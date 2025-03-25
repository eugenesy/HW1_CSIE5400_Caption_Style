# HW1 - Artificial Intelligence Homework

This repository contains the code and environment configuration for HW1. It includes:

- **task1.py:** Evaluation of Image Captioning Models (BLIP and Phi‑4) on the MSCOCO-Test and Flickr30k datasets.
- **task2.py:** MLLM Image Style Transfer (both Text-to-Image and Image-to-Image pipelines).
- **environment.yml:** Conda environment configuration.
- **README.md:** This documentation file.

## Overview

The project is divided into two main tasks:

- **Task 1:** Image Captioning Evaluation  
  Evaluate captioning models on popular datasets. The code supports checkpointing for resuming long-running jobs and computes evaluation metrics like BLEU, ROUGE, and METEOR.

- **Task 2:** MLLM Image Style Transfer  
  Implement image style transfer using a prompt-generation model (Phi‑4) and diffusion models (Stable Diffusion 3 for Text-to-Image and Stable Diffusion v1.5 for Image-to-Image).

## Directory Structure

.
├── task1.py           # Code for Task 1: Image Captioning Evaluation
├── task2.py           # Code for Task 2: Image Style Transfer
├── environment.yml    # Conda environment configuration file
└── README.md          # This file

## Environment Setup

### Using Conda

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already.
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate hw1_env

Using pip

Alternatively, you can install the dependencies manually:

pip install torch torchvision transformers datasets evaluate tqdm diffusers huggingface-hub pillow pandas

Running the Code

Task 1: Image Captioning Evaluation

Execute the evaluation with the following command:

python task1.py --model phi4 --dataset both --split test --num_gpus 1 --output_csv results.csv

Key Parameters:
-	--model: Select between phi4 and blip.
- --dataset: Options include mscoco, flickr, or both.
- --split: Dataset split (e.g., test).
- --num_samples: (Optional) Evaluate a subset of samples.
- --cuda_visible_devices: (Optional) Specify GPU devices.

Task 2: Image Style Transfer

Run the style transfer process using:

python task2.py --token YOUR_HUGGINGFACE_TOKEN --input_path path/to/images --num_gpus 1

Key Parameters:
- --token: Your Hugging Face API token.
- --input_path: Path to the folder or file containing the input images.
- Additional parameters for prompt generation and image synthesis can be viewed with:

python task2.py --help



Additional Notes
- Ensure that you have the necessary CUDA drivers installed for GPU acceleration.
- The code uses checkpointing to save intermediate results, allowing for job resumption.
- This repository is public and does not contain any personal or sensitive information.
