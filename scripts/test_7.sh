#!/usr/bin/env bash
    # Example on Cityscapes
     python -m torch.distributed.launch --nproc_per_node=4 test.py \
        --train_mode Adaptation \
        --source_dataset gtav \
        --target_dataset cityscapes idd mapillary \
        --val_dataset cityscapes idd mapillary \
        --num_classes 7 \
        --rrotate 0 \
        --bs_mult 1 \
        --relax_denom 0.0 \
        --date 0707 \
        --exp test \
        --ckpt ./logs/ \
        --tb_path ./logs/ \
        --snapshot ./pretrained/BARS_da_7.pth \
