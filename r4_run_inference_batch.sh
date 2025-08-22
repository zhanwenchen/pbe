#!/bin/bash

# Example script to run inference with the model
# Usage: ./run_inference.sh

# Check if checkpoint path is provided
# if [ -z "$1" ]; then
#     echo "Error: Please provide the path to the model checkpoint"
#     echo "Usage: ./run_inference.sh <checkpoint_path>"
#     exit 1
# fi

# export FPATH_CHECKPOINT="/home/ubuntu/pbe/./models/20250627T035445_v1/checkpoints/lightning_logs/version_0/checkpoints/epoch=39-step=40000.ckpt/pytorch_model.bin"
# export FPATH_CHECKPOINT="/home/ubuntu/pbe/checkpoints/model.ckpt"
export FPATH_CHECKPOINT="/home/ubuntu/pbe/models/20250714T210450_v1/checkpoints/lightning_logs/version_0/checkpoints/epoch=16-step=2176.ckpt"

# CHECKPOINT_PATH=$1

# Set paths to example images
# DIRPATH_IMAGE_EMPTY="dataset/open-images/images/test/031652557X.png"
# FNSKU="1338883070"
# EMPTY_PATH="~/AFTAI_Intern25_Tote_Diffusion_backup_mine/data/v1/image_tote/${FNSKU}.png"
# DIRPATH_IMAGE_EMPTY="dataset/open-images/fg_removed/train_mine/${FNSKU}.png"
# DIRPATH_IMAGE_EMPTY="dataset/open-images/fg_removed/train_mine"
DIRPATH_IMAGE_EMPTY="examples/image"
# DIRPATH_MASK="/home/ubuntu/AFTAI_Intern25_Tote_Diffusion_backup_mine/data/v1/item_mask"
DIRPATH_MASK="examples/mask"
# DIRPATH_MASK="dataset/open-images/bbox/train_mine/"
# DIRPATH_MASK="dataset/open-images/bbox/test/031652557X.txt"
# DIRPATH_MASK="/home/ubuntu/pbe/dataset/open-images/bbox/train_mine/${FNSKU}.txt"
# DIRPATH_MASK="/home/ubuntu/pbe/dataset/open-images/bbox/train_mine/${FNSKU}.txt"
# DIRPATH_MASK="/home/ubuntu/AFTAI_Intern25_Tote_Diffusion_backup_mine/data/v1/item_mask/${FNSKU}.png"
# REFERENCE_PATH="dataset/open-images/images/test/031652557X.png"
# REFERENCE_PATH="dataset/open-images/fg_removed/train_mine/${FNSKU}.png"
# REFERENCE_PATH="/home/ubuntu/AFTAI_Intern25_Tote_Diffusion_backup_mine/canonical_representation_1338883070_image_3.jpg"
# DIRPATH_CIREPS="/home/ubuntu/AFTAI_Intern25_Tote_Diffusion_backup_mine/data/v2/cirep_afex"
DIRPATH_CIREPS="examples/reference"

# Create output directory
export OUTPUT_DIR="inference_results_pretrained"

if [ -d "${OUTPUT_DIR}" ]; then
    echo "${OUTPUT_DIR}' already exists. Please choose a different output directory to avoid overwriting data."
    exit 1 # Exit with an error code if the assertion fails
fi
mkdir -p $OUTPUT_DIR

# Run inference
python r4_run_inference_batch.py \
    --ckpt "${FPATH_CHECKPOINT}" \
    --config configs/v1.yaml \
    --dirpath_empty "${DIRPATH_IMAGE_EMPTY}" \
    --dirpath_mask "${DIRPATH_MASK}" \
    --dirpath_cireps "${DIRPATH_CIREPS}" \
    --outdir "${OUTPUT_DIR}" \
    --plms \
    --scale 5.0 \
    --ddim_steps 50
