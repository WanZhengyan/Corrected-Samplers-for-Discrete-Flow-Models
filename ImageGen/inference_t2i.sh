#!/bin/bash
CKPT_PATH=./FUDOKI/checkpoints

torchrun --nproc_per_node 1 inference_t2i.py \
    --batch_size 1 \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 8 \
    --output_dir ./output_t2i
