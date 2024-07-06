import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--num_classes', type=int, default=19,
                        help='7 or 19 classes')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_mode', type=str, default='Domain_transfer',
                        help='Domain_transfer / Source_only / Warm_up / Adaptation')
    parser.add_argument('--source_dataset', nargs='*', type=str, default=['gtav'],
                        help='a list of datasets; gtav, cityscapes, idd, mapillary')
    parser.add_argument('--target_dataset', nargs='*', type=str, default=['cityscapes', 'idd', 'mapillary'],
                        help='a list of datasets; gtav, cityscapes, idd, mapillary')
    parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                        help='uniformly sample images across the multiple source domains')
    parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes', 'idd', 'mapillary'],
                        help='a list consists of cityscapes, idd, mapillary')
    parser.add_argument('--cv', type=int, default=0,
                        help='cross-validation split id to use. Default # of splits set to 3 in config')
    parser.add_argument('--class_uniform_pct', type=float, default=0,
                        help='What fraction of images is uniformly sampled')
    parser.add_argument('--class_uniform_tile', type=int, default=1024,
                        help='tile size for class uniform sampling')
    parser.add_argument('--coarse_boost_classes', type=str, default=None,
                        help='use coarse annotations to boost fine data with specific classes')

    parser.add_argument('--img_wt_loss', action='store_true', default=False,
                        help='per-image class-weighted loss')
    parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                        help='class-weighted loss')
    parser.add_argument('--batch_weighting', action='store_true', default=False,
                        help='Batch weighting for class (use nll class weighting using batch stats')

    parser.add_argument('--jointwtborder', action='store_true', default=False,
                        help='Enable boundary label relaxation')
    parser.add_argument('--strict_bdr_cls', type=str, default='',
                        help='Enable boundary label relaxation for specific classes')
    parser.add_argument('--rlx_off_iter', type=int, default=-1,
                        help='Turn off border relaxation after specific epoch count')
    parser.add_argument('--rescale', type=float, default=1.0,
                        help='Warm Restarts new learning rate ratio compared to original lr')
    parser.add_argument('--repoly', type=float, default=1.5,
                        help='Warm Restart new poly exp')

    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use Nvidia Apex AMP')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='parameter used by apex library')

    parser.add_argument('--sgd', action='store_true', default=True)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--amsgrad', action='store_true', default=False)

    parser.add_argument('--freeze_trunk', action='store_true', default=False)
    parser.add_argument('--hardnm', default=0, type=int,
                        help='0 means no aug, 1 means hard negative mining iter 1,' +
                        '2 means hard negative mining iter 2')

    parser.add_argument('--trunk', type=str, default='resnet101',
                        help='trunk model, can be: resnet101 (default), resnet50')
    parser.add_argument('--max_epoch', type=int, default=180)
    parser.add_argument('--max_iter', type=int, default=30000)
    parser.add_argument('--max_cu_epoch', type=int, default=100000,
                        help='Class Uniform Max Epochs')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--crop_nopad', action='store_true', default=False)
    parser.add_argument('--rrotate', type=int,
                        default=0, help='degree of random roate')
    parser.add_argument('--color_aug', type=float,
                        default=0.0, help='level of color augmentation')
    parser.add_argument('--gblur', action='store_true', default=False,
                        help='Use Guassian Blur Augmentation')
    parser.add_argument('--bblur', action='store_true', default=False,
                        help='Use Bilateral Blur Augmentation')
    parser.add_argument('--lr_schedule', type=str, default='poly',
                        help='name of lr schedule: poly')
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='polynomial LR exponent')
    parser.add_argument('--bs_mult', type=int, default=2,
                        help='Batch size for training per gpu')
    parser.add_argument('--bs_mult_val', type=int, default=1,
                        help='Batch size for Validation per gpu')
    parser.add_argument('--crop_size', type=int, default=720,
                        help='training crop size')
    parser.add_argument('--pre_size', type=int, default=None,
                        help='resize image shorter edge to this before augmentation')
    parser.add_argument('--scale_min', type=float, default=0.5,
                        help='dynamically scale training images down to this size')
    parser.add_argument('--scale_max', type=float, default=2.0,
                        help='dynamically scale training images up to this size')
    parser.add_argument('--target_scale_min', type=float, default=0.5,
                        help='dynamically scale training images down to this size')
    parser.add_argument('--target_scale_max', type=float, default=2.0,
                        help='dynamically scale training images up to this size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--init', type=str, default=None)
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--restore_optimizer', action='store_true', default=False)

    parser.add_argument('--city_mode', type=str, default='train',
                        help='experiment directory date name')
    parser.add_argument('--date', type=str, default='default',
                        help='experiment directory date name')
    parser.add_argument('--exp', type=str, default='default',
                        help='experiment directory name')
    parser.add_argument('--tb_tag', type=str, default='',
                        help='add tag to tb dir')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                        help='Save Checkpoint Point')
    parser.add_argument('--tb_path', type=str, default='logs/tb',
                        help='Save Tensorboard Path')
    parser.add_argument('--syncbn', action='store_true', default=True,
                        help='Use Synchronized BN')
    parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                        help='Dump Augmentated Images for sanity check')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help='Minimum testing to verify nothing failed, ' +
                        'Runs code for 1 epoch of train and val')
    parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                        help='Weight Scaling for the losses')
    parser.add_argument('--maxSkip', type=int, default=0,
                        help='Skip x number of  frames of video augmented dataset')
    parser.add_argument('--scf', action='store_true', default=False,
                        help='scale correction factor')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                        help='url used to set up distributed training')

    parser.add_argument('--relax_denom', type=float, default=2.0)
    parser.add_argument('--clusters', type=int, default=50)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--dynamic', action='store_true', default=False)

    parser.add_argument('--image_in', action='store_true', default=False,
                        help='Input Image Instance Norm')

    parser.add_argument('--ema', action='store_true', default=False)
    parser.add_argument('--ema_update_iter', type=int, default=300)    

    parser.add_argument('--rce', action='store_true', default=True)                    

    # Domain Transfer
    parser.add_argument('--DT_source_dataset', nargs='*', type=str, default=[],
                        help='a list of datasets; gtav, cityscapes, idd, mapillary')
    parser.add_argument('--DT_target_dataset', nargs='*', type=str, default=[],
                        help='a list of datasets; gtav, cityscapes, idd, mapillary')
    parser.add_argument('--DT_snapshot', type=str, default=None)
    parser.add_argument('--mtdt_feat_ch', type=int, default=64)

    # AdaptSeg
    parser.add_argument('--discriminator', type=str, default=None,
                        help='base / FPSE')

    # Augmentation
    parser.add_argument('--mix_mode', type=str, default=None,
                        help='uniform / random / class_dist / class_dist_relation')

    # Self training
    parser.add_argument('--self_training', type=str, default=None)
    parser.add_argument('--feat_ch', type=int, default=256)
    parser.add_argument('--curriculum', action='store_true', default=False)
    parser.add_argument('--initial_ratio', type=float, default=0.0)
    parser.add_argument('--incremental_ratio', type=float, default=0.03)

    # Tensorboard visalize
    parser.add_argument('--tensorboard_visualize', action='store_true', default=False)

    # If you have GPU with less than 32G memory, use following option:
    parser.add_argument('--insufficient_gpu_memory', action='store_true', default=False)



    args = parser.parse_args()

    return parser, args