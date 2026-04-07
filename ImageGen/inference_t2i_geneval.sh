#!/bin/bash
# GenEval benchmark inference with all 5 samplers.
# Run from the ImageGen/ directory.
# Requires: geneval_prompt.txt and evaluation_metadata.jsonl in the current directory.

set -e

CKPT_PATH=./FUDOKI/checkpoints

MASTER_ADDR=localhost MASTER_PORT=12345 python inference_t2i_geneval.py \
    --batch_size 1 \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 8 \
    --output_dir ./output_geneval


