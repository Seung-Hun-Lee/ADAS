"""
Dataset setup and loaders
"""
from datasets import cityscapes
from datasets import mapillary
from datasets import synthia
from datasets import kitti
from datasets import camvid
from datasets import bdd100k
from datasets import gtav
from datasets import idd
from datasets import nullloader

from datasets import multi_loader
from datasets.sampler import DistributedSampler

import torchvision.transforms as standard_transforms

import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch


ignore_label = 255
num_classes = None


def get_train_joint_transform(args, dataset, source=False):
    """
    Get train joint transform
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_joint_transform_list, train_joint_transform
    """

    # Geometric image transformations
    train_joint_transform_list = []

    train_joint_transform_list += [
        # joint_transforms.ResizeHeightWidth(args.crop_size),
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                           crop_nopad=args.crop_nopad,
                                           pre_size=args.pre_size,
                                           scale_min=args.scale_min if source else args.target_scale_min,
                                           scale_max=args.scale_max if source else args.target_scale_max,
                                           ignore_index=dataset.ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]

    if args.rrotate > 0:
        train_joint_transform_list += [joint_transforms.RandomRotate(
            degree=args.rrotate,
            ignore_index=dataset.ignore_label)]

    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # return the raw list for class uniform sampling
    return train_joint_transform_list, train_joint_transform


def get_input_transforms(args, dataset):
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """

    # Image appearance transformations
    train_input_transform = []
    val_input_transform = []
    if args.color_aug > 0.0:
        train_input_transform += [standard_transforms.RandomApply([
            standard_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]

    train_input_transform += [
                                standard_transforms.ToTensor()
    ]
    val_input_transform += [
                            standard_transforms.ToTensor()
    ]
    train_input_transform = standard_transforms.Compose(train_input_transform)
    val_input_transform = standard_transforms.Compose(val_input_transform)

    return train_input_transform, val_input_transform

def get_color_geometric_transforms():
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """

    # Image appearance transformations
    color_input_transform = []
    geometric_input_transform = []

    color_input_transform += [standard_transforms.ColorJitter(0.8, 0.8, 0.8, 0.3)]
    color_input_transform += [extended_transforms.RandomGaussianBlur()]

    geometric_input_transform += [standard_transforms.RandomHorizontalFlip(p=1.0)]

    color_input_transform += [
                              standard_transforms.ToTensor()
    ]
    geometric_input_transform += [
                            standard_transforms.ToTensor()
    ]
    color_input_transform = standard_transforms.Compose(color_input_transform)
    geometric_input_transform = standard_transforms.Compose(geometric_input_transform)

    return color_input_transform, geometric_input_transform

def get_target_transforms(args, dataset):
    """
    Get target transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: target_transform, target_train_transform, target_aux_train_transform
    """

    target_transform = extended_transforms.MaskToTensor()
    if args.jointwtborder:
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(
                dataset.ignore_label, args.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    target_aux_train_transform = extended_transforms.MaskToTensor()

    return target_transform, target_train_transform, target_aux_train_transform


def create_extra_val_loader(args, dataset, val_input_transform, target_transform):
    """
    Create extra validation loader
    Args:
        args: input config arguments
        dataset: dataset class object
        val_input_transform: validation input transforms
        target_transform: target transforms

    return: validation loaders
    """
    if dataset == 'cityscapes':
        val_set = cityscapes.CityScapes('fine', 'val', 0,
                                        transform=val_input_transform,
                                        target_transform=target_transform,
                                        cv_split=args.cv,
                                        image_in=args.image_in,
                                        num_classes=args.num_classes)
    elif dataset == 'idd':
        val_set = idd.Idd('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in,
                                        num_classes=args.num_classes)
    elif dataset == 'bdd100k':
        val_set = bdd100k.BDD100K('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in,
                                        num_classes=args.num_classes)
    elif dataset == 'gtav':
        val_set = gtav.GTAV('val', 0,
                            transform=val_input_transform,
                            target_transform=target_transform,
                            cv_split=args.cv,
                            image_in=args.image_in,
                                        num_classes=args.num_classes)
    elif dataset == 'synthia':
        val_set = synthia.Synthia('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in,
                                        num_classes=args.num_classes)
    elif dataset == 'mapillary':
        eval_size = 1536
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]
        val_set = mapillary.Mapillary('semantic', 'val',
                                      joint_transform_list=val_joint_transform_list,
                                      transform=val_input_transform,
                                      target_transform=target_transform,
                                      test=False,
                                        num_classes=args.num_classes)
    elif dataset == 'null_loader':
        val_set = nullloader.nullloader(args.crop_size)
    else:
        raise Exception('Dataset {} is not supported'.format(dataset))

    if args.syncbn:
        from datasets.sampler import DistributedSampler
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)

    else:
        val_sampler = None

    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False,
                            sampler = val_sampler)
    return val_loader

def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """
    global num_classes
    num_classes = args.num_classes
    args.train_batch_size = args.bs_mult * args.ngpu
    if args.bs_mult_val > 0:
        args.val_batch_size = args.bs_mult_val * args.ngpu
    else:
        args.val_batch_size = args.bs_mult * args.ngpu

    # Readjust batch size to mini-batch size for syncbn
    if args.syncbn:
        args.train_batch_size = args.bs_mult
        args.val_batch_size = args.bs_mult_val

    args.num_workers = 8 #1 * args.ngpu
    if args.test_mode:
        args.num_workers = 1

    source_train_sets = []
    target_train_sets = []

    if 'gtav' in args.source_dataset:
        dataset = gtav
        gtav_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset, source=True)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.GTAVUniform(
                gtav_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in, num_classes=args.num_classes)
        else:
            train_set = gtav.GTAV(
                gtav_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in, num_classes=args.num_classes)

        source_train_sets.append(train_set)
    
    elif 'gtav' in args.target_dataset:
        dataset = gtav
        gtav_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        train_set = gtav.GTAV(
                gtav_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in, num_classes=args.num_classes)
        target_train_sets.append(train_set)

    if 'cityscapes' in args.source_dataset:
        dataset = cityscapes
        city_mode = args.city_mode #'train' ## Can be trainval
        city_quality = 'fine'
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset, source=True)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.CityScapesUniform(
                city_quality, city_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in, num_classes=args.num_classes)
        else:
            train_set = dataset.CityScapes(
                city_quality, city_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                image_in=args.image_in, num_classes=args.num_classes)

        source_train_sets.append(train_set)

    elif 'cityscapes' in args.target_dataset:
        dataset = cityscapes
        city_mode = args.city_mode #'train' ## Can be trainval
        city_quality = 'fine'
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        train_set = dataset.CityScapes(
                city_quality, city_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                image_in=args.image_in, num_classes=args.num_classes)

        target_train_sets.append(train_set)

    if 'bdd100k' in args.source_dataset:
        dataset = bdd100k
        bdd_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset, source=True)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.BDD100KUniform(
                bdd_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in, num_classes=args.num_classes)
        else:
            train_set = dataset.BDD100K(
                bdd_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in, num_classes=args.num_classes)

        source_train_sets.append(train_set)

    elif 'bdd100k' in args.target_dataset:
        dataset = bdd100k
        bdd_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        train_set = dataset.BDD100K(
                bdd_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in, num_classes=args.num_classes)

        target_train_sets.append(train_set)


    if 'idd' in args.source_dataset:
        dataset = idd
        idd_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset, source=True)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.IddUniform(
                idd_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in, num_classes=args.num_classes)
        else:
            train_set = dataset.Idd(
                idd_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in, num_classes=args.num_classes)

        source_train_sets.append(train_set)

    elif 'idd' in args.target_dataset:
        dataset = idd
        idd_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        

        train_set = dataset.Idd(
                idd_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in, num_classes=args.num_classes)

        target_train_sets.append(train_set)

   
    if 'synthia' in args.source_dataset:
        dataset = synthia
        synthia_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset, source=True)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None

            train_set = dataset.SynthiaUniform(
                synthia_mode, args.maxSkip,
                joint_transform_list=train_joint_transform_list,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                class_uniform_pct=args.class_uniform_pct,
                class_uniform_tile=args.class_uniform_tile,
                test=args.test_mode,
                coarse_boost_classes=coarse_boost_classes,
                image_in=args.image_in)
        else:
            train_set = dataset.Synthia(
                synthia_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in)

        source_train_sets.append(train_set)

    elif 'synthia' in args.target_dataset:
        dataset = synthia
        synthia_mode = 'train' ## Can be trainval
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        train_set = dataset.Synthia(
                synthia_mode, 0,
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                target_aux_transform=target_aux_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv,
                image_in=args.image_in)

        target_train_sets.append(train_set)


    if 'mapillary' in args.source_dataset:
        dataset = mapillary
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset, source=True)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)



        train_set = dataset.Mapillary(
            'semantic', 'train',
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            target_aux_transform=target_aux_train_transform,
            image_in=args.image_in,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_tile=args.class_uniform_tile,
            test=args.test_mode, num_classes=args.num_classes)
        source_train_sets.append(train_set)

    elif 'mapillary' in args.target_dataset:
        dataset = mapillary
        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args, dataset)
        train_input_transform, val_input_transform = get_input_transforms(args, dataset)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)

        height_size = 1536
        height_resize_list = [
            joint_transforms.ResizeHeight(height_size),
            joint_transforms.CenterCropPad(height_size)]
        
        train_joint_transform_list = height_resize_list + train_joint_transform_list

        train_set = dataset.Mapillary(
            'semantic', 'train',
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            target_aux_transform=target_aux_train_transform,
            image_in=args.image_in,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=0,
            class_uniform_tile=args.class_uniform_tile,
            test=args.test_mode, num_classes=args.num_classes)
        target_train_sets.append(train_set)


    if 'null_loader' in args.source_dataset:
        train_set = nullloader.nullloader(args.crop_size)
        val_set = nullloader.nullloader(args.crop_size)

        source_train_sets.append(train_set)

    if len(source_train_sets) == 0:
        raise Exception('Dataset {} is not supported'.format(args.source_dataset))
    
    if len(target_train_sets) == 0:
        raise Exception('Dataset {} is not supported'.format(args.target_dataset))

    if len(source_train_sets) != len(args.source_dataset):
        raise Exception('Something went wrong. Please check your dataset names are valid')
    
    if len(target_train_sets) != len(args.target_dataset):
        raise Exception('Something went wrong. Please check your dataset names are valid')

    # Define new train data set that has all the train sets
    # Define new val data set that has all the val sets

    if len(args.source_dataset) != 1:
        if args.image_uniform_sampling:
            source_train_set = ConcatDataset(source_train_sets)
        else:
            source_train_set = multi_loader.DomainUniformConcatDataset(args, source_train_sets)
    else:
        source_train_set = source_train_sets[0]

    if len(args.target_dataset) != 1:
        if args.image_uniform_sampling:
            target_train_set = ConcatDataset(target_train_sets)
        else:
            target_train_set = multi_loader.DomainUniformConcatDataset(args, target_train_sets)
    else:
        target_train_set = target_train_sets[0]

    if args.syncbn:
        source_train_sampler = DistributedSampler(source_train_set, pad=True, permutation=True, consecutive_sample=False)
        target_train_sampler = DistributedSampler(target_train_set, pad=True, permutation=True, consecutive_sample=False)
    else:
        source_train_sampler = None
        target_train_sampler = None

    source_train_loader = DataLoader(source_train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(source_train_sampler is None), drop_last=True, sampler = source_train_sampler)
    target_train_loader = DataLoader(target_train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(target_train_sampler is None), drop_last=True, sampler = target_train_sampler)

    extra_val_loader = {}
    for val_dataset in args.val_dataset:
        extra_val_loader[val_dataset] = create_extra_val_loader(args, val_dataset, val_input_transform, target_transform)

    return source_train_loader, target_train_loader, source_train_set, target_train_set, extra_val_loader

