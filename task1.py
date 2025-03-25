#!/usr/bin/env python3
import os
import pickle
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import argparse

from transformers import (
    BlipForConditionalGeneration, BlipProcessor,
    AutoModelForCausalLM, AutoProcessor, GenerationConfig
)
from datasets import load_dataset
import evaluate

#######################################
# Checkpointing functions (per worker)
#######################################

def save_checkpoint_local(predictions, references, start_index, checkpoint_file):
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            "predictions": predictions,
            "references": references,
            "start_index": start_index
        }, f)

def load_checkpoint_local(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    else:
        return None

#######################################
# Worker function for process-level evaluation
#######################################

def worker(gpu_id, indices, dataset_split, caption_fn, return_dict, checkpoint_prefix, checkpoint_interval):
    torch.cuda.set_device(gpu_id)
    checkpoint_file = f"{checkpoint_prefix}_{gpu_id}.pkl"
    checkpoint = load_checkpoint_local(checkpoint_file)
    start_index_local = checkpoint["start_index"] if checkpoint else 0
    predictions_local = checkpoint["predictions"] if checkpoint else []
    references_local = checkpoint["references"] if checkpoint else []

    # Reinitialize the model on this GPU based on caption_fn name.
    if caption_fn.__name__ == "generate_caption_blip":
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
        def local_caption(image):
            inputs = blip_processor(image, return_tensors="pt").to("cuda")
            outputs = blip_model.generate(**inputs)
            caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        caption_fn = local_caption

    elif caption_fn.__name__ == "generate_caption_phi4":
        phi4_model_path = "microsoft/Phi-4-multimodal-instruct"
        phi4_processor = AutoProcessor.from_pretrained(phi4_model_path, trust_remote_code=True)
        phi4_model = AutoModelForCausalLM.from_pretrained(
            phi4_model_path,
            device_map=None,  # Disable auto device mapping
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation='eager',
        ).to(f"cuda:{gpu_id}")  # Move model to assigned GPU
        phi4_generation_config = GenerationConfig.from_pretrained(phi4_model_path)
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        def local_caption(image):
            prompt = (f"{user_prompt}<|image_1|>A detailed description of the image, "
                      f"Please provide just the caption."
                      f"{prompt_suffix}{assistant_prompt}")
            inputs = phi4_processor(images=image, text=prompt, return_tensors="pt").to(f"cuda:{gpu_id}")  
            outputs = phi4_model.generate(
                **inputs,
                generation_config=phi4_generation_config,
                max_new_tokens=20
            ).to(f"cuda:{gpu_id}")  
            caption = phi4_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            caption = caption.replace("A detailed description of the image, Please provide just the caption.", "").strip()
            return caption
        caption_fn = local_caption

    for local_i in tqdm(range(start_index_local, len(indices)), desc=f"GPU {gpu_id} Evaluating", unit="sample"):
        idx = indices[local_i]
        example = dataset_split[idx]
        image = example["image"]
        # Get reference caption (assumes field "caption")
        ref_list = example.get("caption", ["No reference"])
        ref = ref_list if isinstance(ref_list, list) and len(ref_list) > 0 else "No reference"
        try:
            pred = caption_fn(image)
        except Exception as e:
            pred = f"Error: {e}"
        predictions_local.append(pred)
        references_local.append(ref)

        # Save checkpoint periodically
        if (local_i + 1) % checkpoint_interval == 0:
            save_checkpoint_local(predictions_local, references_local, local_i + 1, checkpoint_file)

    # Save final checkpoint for this worker
    save_checkpoint_local(predictions_local, references_local, len(indices), checkpoint_file)
    return_dict[gpu_id] = {
        "predictions": predictions_local,
        "references": references_local
    }

#######################################
# Metric computation function
#######################################

def compute_metrics(predictions, references):
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bleu = bleu_metric.compute(predictions=predictions, references=references)
    rouge = rouge_metric.compute(predictions=predictions, references=references)
    meteor = meteor_metric.compute(predictions=predictions, references=references)
    return {
        "BLEU": bleu["bleu"],
        "ROUGE-1": rouge["rouge1"],
        "ROUGE-2": rouge["rouge2"],
        "METEOR": meteor["meteor"]
    }

#######################################
# Parallel evaluation function
#######################################

def evaluate_model_parallel(dataset, split, caption_fn, num_samples, checkpoint_prefix, num_gpus, checkpoint_interval):
    dataset_split = dataset[split] if num_samples is None else dataset[split].select(range(num_samples))
    total_samples = len(dataset_split)

    # Split indices evenly among GPUs
    indices = list(range(total_samples))
    chunk_size = total_samples // num_gpus
    indices_chunks = [indices[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]
    if total_samples % num_gpus != 0:
        indices_chunks[-1].extend(indices[num_gpus * chunk_size:])

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker,
            args=(gpu_id, indices_chunks[gpu_id], dataset_split, caption_fn, return_dict, checkpoint_prefix, checkpoint_interval)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Aggregate results from all processes
    all_predictions = []
    all_references = []
    for gpu_id in range(num_gpus):
        res = return_dict[gpu_id]
        all_predictions.extend(res["predictions"])
        all_references.extend(res["references"])

    metrics = compute_metrics(all_predictions, all_references)
    return metrics, all_predictions, all_references

#######################################
# Dummy caption generation functions
#######################################

def generate_caption_blip(image):
    """
    Dummy placeholder for process-level parallelism.
    In the worker, this function’s name is used to decide which model to load.
    """
    pass

def generate_caption_phi4(image):
    """
    Dummy placeholder for process-level parallelism.
    In the worker, this function’s name is used to decide which model to load.
    """
    pass

#######################################
# Dataset loading function
#######################################

def load_datasets():
    mscoco = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")
    flickr30k = load_dataset("nlphuji/flickr30k")
    return mscoco, flickr30k

#######################################
# Main execution function
#######################################

def main(args):
    # Set CUDA_VISIBLE_DEVICES environment variable if provided
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    mp.set_start_method(args.mp_start_method, force=True)
    mscoco, flickr30k = load_datasets()

    results = []
    # Choose caption function based on user input.
    if args.model.lower() == "phi4":
        caption_fn = generate_caption_phi4
        model_name = "Phi‑4"
    elif args.model.lower() == "blip":
        caption_fn = generate_caption_blip
        model_name = "BLIP"
    else:
        raise ValueError("Invalid model specified. Choose either 'phi4' or 'blip'.")

    # Evaluate based on selected dataset
    if args.dataset.lower() == "mscoco":
        print(f"Evaluating {model_name} on MSCOCO-Test:")
        metrics, _, _ = evaluate_model_parallel(
            mscoco, args.split, caption_fn, args.num_samples, args.checkpoint_prefix + "_mscoco", args.num_gpus, args.checkpoint_interval
        )
        print("Metrics:", metrics)
        results.append({
            "Model": model_name,
            "Dataset": "MSCOCO-Test",
            "BLEU": metrics["BLEU"],
            "ROUGE-1": metrics["ROUGE-1"],
            "ROUGE-2": metrics["ROUGE-2"],
            "METEOR": metrics["METEOR"]
        })

    elif args.dataset.lower() == "flickr":
        print(f"Evaluating {model_name} on Flickr30k:")
        metrics, _, _ = evaluate_model_parallel(
            flickr30k, args.split, caption_fn, args.num_samples, args.checkpoint_prefix + "_flickr", args.num_gpus, args.checkpoint_interval
        )
        print("Metrics:", metrics)
        results.append({
            "Model": model_name,
            "Dataset": "Flickr30k",
            "BLEU": metrics["BLEU"],
            "ROUGE-1": metrics["ROUGE-1"],
            "ROUGE-2": metrics["ROUGE-2"],
            "METEOR": metrics["METEOR"]
        })

    elif args.dataset.lower() == "both":
        print(f"Evaluating {model_name} on MSCOCO-Test:")
        metrics_mscoco, _, _ = evaluate_model_parallel(
            mscoco, args.split, caption_fn, args.num_samples, args.checkpoint_prefix + "_mscoco", args.num_gpus, args.checkpoint_interval
        )
        print("Metrics:", metrics_mscoco)
        results.append({
            "Model": model_name,
            "Dataset": "MSCOCO-Test",
            "BLEU": metrics_mscoco["BLEU"],
            "ROUGE-1": metrics_mscoco["ROUGE-1"],
            "ROUGE-2": metrics_mscoco["ROUGE-2"],
            "METEOR": metrics_mscoco["METEOR"]
        })

        print(f"Evaluating {model_name} on Flickr30k:")
        metrics_flickr, _, _ = evaluate_model_parallel(
            flickr30k, args.split, caption_fn, args.num_samples, args.checkpoint_prefix + "_flickr", args.num_gpus, args.checkpoint_interval
        )
        print("Metrics:", metrics_flickr)
        results.append({
            "Model": model_name,
            "Dataset": "Flickr30k",
            "BLEU": metrics_flickr["BLEU"],
            "ROUGE-1": metrics_flickr["ROUGE-1"],
            "ROUGE-2": metrics_flickr["ROUGE-2"],
            "METEOR": metrics_flickr["METEOR"]
        })
    else:
        raise ValueError("Invalid dataset specified. Choose 'mscoco', 'flickr', or 'both'.")

    results_df = pd.DataFrame(results)
    print("Summary of results:")
    print(results_df)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

#######################################
# CLI Argument Parsing
#######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parallel Evaluation of Image Captioning Models")
    parser.add_argument("--cuda_visible_devices", type=str, default="0", 
                        help="CUDA_VISIBLE_DEVICES environment variable (default: '0')")
    parser.add_argument("--mp_start_method", type=str, default="fork", 
                        help="Multiprocessing start method ('fork' or 'spawn', default: 'fork')")
    parser.add_argument("--model", type=str, default="phi4", choices=["phi4", "blip"], 
                        help="Model to evaluate: 'phi4' or 'blip' (default: 'phi4')")
    parser.add_argument("--dataset", type=str, default="both", choices=["mscoco", "flickr", "both"], 
                        help="Dataset to evaluate on (default: 'both')")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use for evaluation (default: 'test')")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to evaluate (default: all samples)")
    parser.add_argument("--checkpoint_prefix", type=str, default="checkpoint", 
                        help="Prefix for checkpoint files (default: 'checkpoint')")
    parser.add_argument("--num_gpus", type=int, default=1, 
                        help="Number of GPUs to use (default: 1)")
    parser.add_argument("--checkpoint_interval", type=int, default=100, 
                        help="Interval for saving checkpoints (default: 100)")
    parser.add_argument("--output_csv", type=str, default="results.csv", 
                        help="CSV file to save results (default: 'results.csv')")

    args = parser.parse_args()
    main(args)