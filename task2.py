import os
import pickle
import glob
import argparse
from PIL import Image
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from diffusers import StableDiffusion3Pipeline, AutoPipelineForImage2Image
from diffusers.utils import load_image
from huggingface_hub import login

# Set GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#######################################
# Utility: Checkpointing Functions
#######################################
def save_checkpoint(outputs, start_index, checkpoint_file):
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({"outputs": outputs, "start_index": start_index}, f)

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

#######################################
# Phase 1: Prompt Generation (Phi‑4)
#######################################
def worker_prompt(gpu_id, indices, file_list, checkpoint_prefix, checkpoint_interval,
                  phi4_max_new_tokens, phi4_model_path, phi4_base_instruction):
    torch.cuda.set_device(gpu_id)
    phi4_processor = AutoProcessor.from_pretrained(phi4_model_path, trust_remote_code=True)
    phi4_model = AutoModelForCausalLM.from_pretrained(
        phi4_model_path,
        device_map=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        _attn_implementation='eager'
    ).to(f"cuda:{gpu_id}")
    phi4_gen_config = GenerationConfig.from_pretrained(phi4_model_path)

    checkpoint_file = f"{checkpoint_prefix}_{gpu_id}.pkl"
    checkpoint = load_checkpoint(checkpoint_file)
    start_index_local = checkpoint["start_index"] if checkpoint else 0
    outputs = checkpoint["outputs"] if checkpoint else []

    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"
    input_prompt_template = f"{user_prompt}<|image_1|>{phi4_base_instruction} {prompt_suffix}{assistant_prompt}"
    
    for i in tqdm(range(start_index_local, len(indices)), desc=f"GPU {gpu_id} (Prompt)", unit="sample"):
        idx = indices[i]
        file_path = file_list[idx]
        try:
            content_image = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {file_path}: {e}")
            continue

        inputs = phi4_processor(images=content_image, text=input_prompt_template, return_tensors="pt")
        inputs = {k: (v.to(f"cuda:{gpu_id}") if v is not None else None) for k, v in inputs.items()}
        gen_ids = phi4_model.generate(**inputs, generation_config=phi4_gen_config, max_new_tokens=phi4_max_new_tokens)
        generated_text = phi4_processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        generated_text = generated_text.replace(phi4_base_instruction, "").strip()
        outputs.append((file_path, generated_text))
        
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(outputs, i + 1, checkpoint_file)
    save_checkpoint(outputs, len(indices), checkpoint_file)
    return outputs

def worker_wrapper_prompt(gpu_id, indices, file_list, checkpoint_prefix, checkpoint_interval,
                          phi4_max_new_tokens, phi4_model_path, phi4_base_instruction, return_dict):
    try:
        res = worker_prompt(gpu_id, indices, file_list, checkpoint_prefix, checkpoint_interval,
                            phi4_max_new_tokens, phi4_model_path, phi4_base_instruction)
    except Exception as e:
        print(f"Error in GPU {gpu_id} prompt worker: {e}")
        res = []
    return_dict[gpu_id] = res

def parallel_process_prompts(file_list, checkpoint_prefix, num_gpus, checkpoint_interval,
                             phi4_max_new_tokens, phi4_model_path, phi4_base_instruction):
    total_samples = len(file_list)
    indices = list(range(total_samples))
    chunk_size = total_samples // num_gpus
    indices_chunks = [indices[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]
    if total_samples % num_gpus != 0:
        indices_chunks[-1].extend(indices[num_gpus * chunk_size:])
    
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_wrapper_prompt, args=(
            gpu_id, indices_chunks[gpu_id], file_list, checkpoint_prefix, checkpoint_interval,
            phi4_max_new_tokens, phi4_model_path, phi4_base_instruction, return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    outputs = []
    for gpu_id in range(num_gpus):
        outputs.extend(return_dict.get(gpu_id, []))
    return outputs

#######################################
# Phase 2: Image Generation (Stable Diffusion 3)
#######################################
def worker_t2i(gpu_id, indices, prompts_list, checkpoint_prefix, checkpoint_interval,
               sd3_inference_steps, sd3_guidance_scale, sd3_negative_prompt, sd3_output_dir):
    torch.cuda.set_device(gpu_id)
    sd_pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    ).to(f"cuda:{gpu_id}")
    os.makedirs(sd3_output_dir, exist_ok=True)
    
    checkpoint_file = f"{checkpoint_prefix}_{gpu_id}.pkl"
    checkpoint = load_checkpoint(checkpoint_file)
    start_index_local = checkpoint["start_index"] if checkpoint else 0
    outputs = checkpoint["outputs"] if checkpoint else []
    
    for i in tqdm(range(start_index_local, len(indices)), desc=f"GPU {gpu_id} (Image)", unit="sample"):
        idx = indices[i]
        file_path, prompt = prompts_list[idx]
        try:
            print(prompt)
            result = sd_pipeline(
                prompt,
                num_inference_steps=sd3_inference_steps,
                guidance_scale=sd3_guidance_scale,
                negative_prompt=sd3_negative_prompt
            )
            stylized_image = result.images[0]
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}' from {file_path}: {e}")
            continue
        
        stylized_image = stylized_image.resize((224, 224))
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(sd3_output_dir, f"{name}.jpg")
        stylized_image.save(output_path)
        outputs.append((file_path, prompt, output_path))
        
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(outputs, i + 1, checkpoint_file)
    save_checkpoint(outputs, len(indices), checkpoint_file)
    return outputs

def worker_wrapper_t2i(gpu_id, indices, prompts, checkpoint_prefix, checkpoint_interval,
                       sd3_inference_steps, sd3_guidance_scale, sd3_negative_prompt, sd3_output_dir, return_dict):
    try:
        res = worker_t2i(gpu_id, indices, prompts, checkpoint_prefix, checkpoint_interval,
                         sd3_inference_steps, sd3_guidance_scale, sd3_negative_prompt, sd3_output_dir)
    except Exception as e:
        print(f"Error in GPU {gpu_id} image worker: {e}")
        res = []
    return_dict[gpu_id] = res

def parallel_process_images(prompts, checkpoint_prefix, num_gpus, checkpoint_interval,
                            sd3_inference_steps, sd3_guidance_scale, sd3_negative_prompt, sd3_output_dir):
    total_samples = len(prompts)
    indices = list(range(total_samples))
    chunk_size = total_samples // num_gpus
    indices_chunks = [indices[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]
    if total_samples % num_gpus != 0:
        indices_chunks[-1].extend(indices[num_gpus * chunk_size:])
    
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_wrapper_t2i, args=(
            gpu_id, indices_chunks[gpu_id], prompts, checkpoint_prefix, checkpoint_interval,
            sd3_inference_steps, sd3_guidance_scale, sd3_negative_prompt, sd3_output_dir, return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    outputs = []
    for gpu_id in range(num_gpus):
        outputs.extend(return_dict.get(gpu_id, []))
    return outputs

#######################################
# Phase 3: Image-to-Image Pipeline (Stable Diffusion v1.5)
#######################################
def run_image_to_image_pipeline(prompts_pkl, sd15_inference_steps, sd15_guidance_scale,
                                sd15_strength, sd15_negative_prompt, sd15_output_dir,
                                sd15_resize_width, sd15_resize_height):
    with open(prompts_pkl, "rb") as f:
        prompts_data = pickle.load(f)
    os.makedirs(sd15_output_dir, exist_ok=True)
    
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline.to("cuda")
    pipeline.enable_xformers_memory_efficient_attention()
    
    for content_image_path, prompt in prompts_data:
        try:
            content_image = Image.open(content_image_path).convert("RGB")
            content_image = content_image.resize((sd15_resize_width, sd15_resize_height), resample=Image.BICUBIC)
            init_image = load_image(content_image)
        except Exception as e:
            print(f"Error processing image {content_image_path}: {e}")
            continue
        
        print(prompt)

        result = pipeline(
            prompt,
            image=init_image,
            num_inference_steps=sd15_inference_steps,
            guidance_scale=sd15_guidance_scale,
            strength=sd15_strength,
            negative_prompt=sd15_negative_prompt
        )
        stylized_image = result.images[0]
        base_name = os.path.basename(content_image_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(sd15_output_dir, f"{name}_task2_2_gridsearched.jpg")
        stylized_image.save(output_path)
        print(f"Saved stylized image for {content_image_path} to {output_path}")

#######################################
# Main Execution Block
#######################################
def run_pipeline(args):
    # Determine if input_path is a directory or a single file
    if os.path.isdir(args.input_path):
        file_list = sorted(glob.glob(os.path.join(args.input_path, "*.*")))
    elif os.path.isfile(args.input_path):
        file_list = [args.input_path]
    else:
        raise ValueError("Provided input_path is neither a directory nor a valid file.")
    
    print("Starting prompt generation phase using Phi‑4...")
    prompt_outputs = parallel_process_prompts(
        file_list,
        checkpoint_prefix=args.checkpoint_prefix_prompts,
        num_gpus=args.num_gpus,
        checkpoint_interval=args.phi4_checkpoint_interval,
        phi4_max_new_tokens=args.phi4_max_new_tokens,
        phi4_model_path=args.phi4_model_path,
        phi4_base_instruction=args.phi4_base_instruction
    )
    
    with open(args.prompts_pkl, "wb") as f:
        pickle.dump(prompt_outputs, f)
    print("Prompt generation complete.")
    
    print("Starting image generation phase using Stable Diffusion 3...")
    _ = parallel_process_images(
        prompt_outputs,
        checkpoint_prefix=args.checkpoint_prefix_images,
        num_gpus=args.num_gpus,
        checkpoint_interval=args.sd3_checkpoint_interval,
        sd3_inference_steps=args.sd3_inference_steps,
        sd3_guidance_scale=args.sd3_guidance_scale,
        sd3_negative_prompt=args.sd3_negative_prompt,
        sd3_output_dir=args.sd3_output_dir
    )
    print("Image generation complete. Stylized images saved in the designated folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU image processing pipeline with extensive parameter control.")
    # Hugging Face token argument
    parser.add_argument("--token", type=str, required=True,
                        help="Hugging Face API token with access to stabilityai/stable-diffusion-3-medium-diffusers")
    # Input and GPU settings
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to a folder of images or a single image file.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for processing.")
    
    # Prompt generation (Phi‑4) parameters
    parser.add_argument("--phi4_model_path", type=str, default="microsoft/Phi-4-multimodal-instruct",
                        help="Model path for Phi‑4 prompt generation.")
    parser.add_argument("--phi4_max_new_tokens", type=int, default=50,
                        help="Max new tokens to generate with Phi‑4.")
    parser.add_argument("--phi4_checkpoint_interval", type=int, default=10,
                        help="Checkpoint interval for prompt generation.")
    parser.add_argument("--phi4_base_instruction", type=str,
                        default=("Describe an image and Start it with 'A Charles M. Schulz’s Charlie Brown kid Peanut comic style of "
                                 "[long detailed description of the image or face]'"),
                        help="Base instruction for the Phi‑4 prompt generation.")
    parser.add_argument("--checkpoint_prefix_prompts", type=str, default="checkpoint_prompts",
                        help="Checkpoint file prefix for prompt generation.")
    
    # Image generation (Stable Diffusion 3) parameters
    parser.add_argument("--sd3_inference_steps", type=int, default=30,
                        help="Number of inference steps for SD3.")
    parser.add_argument("--sd3_guidance_scale", type=float, default=7.0,
                        help="Guidance scale for SD3.")
    parser.add_argument("--sd3_checkpoint_interval", type=int, default=10,
                        help="Checkpoint interval for SD3.")
    parser.add_argument("--sd3_negative_prompt", type=str,
                        default=("photorealistic, hyper-realistic, 3D render, digital painting, oil painting, extra limbs, "
                                 "deformed features, complex background, watermark, signature"),
                        help="Negative prompt for SD3.")
    parser.add_argument("--sd3_output_dir", type=str, default="t2i_results",
                        help="Output directory for SD3 generated images.")
    parser.add_argument("--checkpoint_prefix_images", type=str, default="checkpoint_images",
                        help="Checkpoint file prefix for SD3 image generation.")
    
    # Image-to-Image (Stable Diffusion v1.5) parameters
    parser.add_argument("--sd15_inference_steps", type=int, default=100,
                        help="Number of inference steps for SD1.5 image-to-image.")
    parser.add_argument("--sd15_guidance_scale", type=float, default=6,
                        help="Guidance scale for SD1.5 image-to-image.")
    parser.add_argument("--sd15_strength", type=float, default=0.8,
                        help="Strength scale for SD1.5 image-to-image.")
    parser.add_argument("--sd15_negative_prompt", type=str,
                        default=("real human, photorealistic, hyper-realistic, 3D render, digital painting, oil painting, extra limbs, "
                                 "deformed features, complex background, watermark, signature"),
                        help="Negative prompt for SD1.5 image-to-image.")
    parser.add_argument("--sd15_output_dir", type=str, default="i2i_results",
                        help="Output directory for SD1.5 image-to-image results.")
    parser.add_argument("--sd15_resize_width", type=int, default=512,
                        help="Resize width for SD1.5 input image.")
    parser.add_argument("--sd15_resize_height", type=int, default=512,
                        help="Resize height for SD1.5 input image.")
    
    # Prompts pickle file name
    parser.add_argument("--prompts_pkl", type=str, default="prompts.pkl",
                        help="Filename for saving generated prompts.")
    
    args = parser.parse_args()
    
    # Login using the provided token
    login(token=args.token)
    
    mp.set_start_method("spawn")  # Change to 'spawn' on Windows if necessary
    run_pipeline(args)
    run_image_to_image_pipeline(
        prompts_pkl=args.prompts_pkl,
        sd15_inference_steps=args.sd15_inference_steps,
        sd15_guidance_scale=args.sd15_guidance_scale,
        sd15_strength=args.sd15_strength,
        sd15_negative_prompt=args.sd15_negative_prompt,
        sd15_output_dir=args.sd15_output_dir,
        sd15_resize_width=args.sd15_resize_width,
        sd15_resize_height=args.sd15_resize_height
    )