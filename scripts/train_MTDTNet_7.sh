python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --train_mode Domain_transfer \
    --source_dataset gtav \
    --target_dataset cityscapes idd mapillary \
    --DT_source_dataset gtav \
    --DT_target_dataset cityscapes idd mapillary \
    --num_classes 7 \
    --city_mode 'train' \
    --lr_schedule poly \
    --lr 0.01 \
    --poly_exp 0.9 \
    --max_cu_epoch 10000 \
    --class_uniform_pct 0.5 \
    --crop_size 768 \
    --scale_min 0.5 \
    --scale_max 2.0 \
    --target_scale_min 0.5 \
    --target_scale_max 1.2 \
    --rrotate 0 \
    --max_iter 3000 \
    --bs_mult 1 \
    --color_aug 0 \
    --relax_denom 0.0 \
    --date 0707 \
    --exp MTDTNet_7 \
    --ckpt ./logs/ \
    --tb_path ./logs/ \
    # --tensorboard_visualize
