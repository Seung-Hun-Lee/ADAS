"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random
import copy
from network.deeplabv2 import Deeplab

from args import get_args


class ADAS_Trainer:
    def __init__(self):
        # Argument Parser
        parser, args = get_args()
        self.args = args

        self.args.world_size = 1

        if 'WORLD_SIZE' in os.environ:
            # self.args.apex = int(os.environ['WORLD_SIZE']) > 1
            self.args.world_size = int(os.environ['WORLD_SIZE'])
            print("Total world size: ", int(os.environ['WORLD_SIZE']))

        torch.cuda.set_device(self.args.local_rank)
        print('My Rank:', self.args.local_rank)
        # Initialize distributed communication
        self.args.dist_url = self.args.dist_url + str(8000 + (int(time.time()%1000))//10)

        torch.distributed.init_process_group(backend='nccl',
                                            init_method=self.args.dist_url,
                                            world_size=self.args.world_size,
                                            rank=self.args.local_rank)
        
        # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
        assert_and_infer_cfg(self.args)
        self.writer = prep_experiment(self.args, parser)
        if self.args.tensorboard_visualize:
            from tensorboard_visualizer import Tensorboard_Visualizer
            self.visualizer = Tensorboard_Visualizer(self.args, self.writer)

        self.source_train_loader, self.target_train_loader, self.source_train_obj, \
        self.target_train_obj, self.extra_val_loaders = datasets.setup_loaders(self.args)

        self.criterion, self.criterion_val = loss.get_loss(args)

        self.time_meter = AverageMeter()

        self.epoch = 1
        self.i = 1

        if self.args.train_mode == 'Source_only' or self.args.train_mode == 'Domain_transfer':
            self.iter_per_epoch = len(self.source_train_loader)
        else:
            self.iter_per_epoch = len(self.target_train_loader)

        self.losses = dict()
        if self.args.train_mode == 'Source_only':
            self.losses['src'] = AverageMeter()
        elif self.args.train_mode == 'Domain_transfer':
            self.losses['recon'] = AverageMeter()
            self.losses['adv'] = AverageMeter()
            self.losses['per'] = AverageMeter()
            self.losses['dis'] = AverageMeter()
        elif self.args.train_mode == 'Warm_up':
            self.losses['src'] = AverageMeter()
            self.losses['tgt'] = AverageMeter()
            self.losses['dis'] = AverageMeter()
        else:
            self.losses['src'] = AverageMeter()
            self.losses['tgt'] = AverageMeter()

        self.best_miou, self.curr_miou = dict(), dict()
        for dset in self.args.val_dataset:
            self.best_miou[dset], self.curr_miou[dset] = 0, 0

        if self.args.self_training == 'bars':
            from network.bars import Bidirectional_Adaptive_Region_Selection
            self.bars = Bidirectional_Adaptive_Region_Selection(self.args)
        if self.args.mix_mode is not None:
            from network.domain_mixer import Domain_Mixer
            self.mixer = Domain_Mixer(self.args)

        self.set_network()
        self.set_optimizer()
        self.load_networks()
        if args.ema:
            self.create_ema_model()
            self.ema_model.eval()

    def set_network(self):
        self.net = dict()

        if self.args.train_mode != 'Domain_transfer':
            self.net['T'] = Deeplab(self.args, num_classes=self.args.num_classes, criterion=self.criterion, 
                                 initialization=self.args.init, freeze_bn=(self.args.train_mode != 'Source_only'))
            self.net['T'] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net['T'])

        if len(self.args.DT_source_dataset) != 0:
            from network.MTDTNet import (Multi_Head_Discriminator, Multi_Target_Domain_Transfer, VGG19)
            self.net['DT'] = Multi_Target_Domain_Transfer(self.args)
            if self.args.train_mode == 'Domain_transfer':
                self.net['D'] = Multi_Head_Discriminator(self.args)
                self.net['P'] = VGG19()

        if self.args.discriminator == 'base':
            from network.Discriminator import FCDiscriminator
            self.net['D'] = FCDiscriminator(self.args)
        elif self.args.discriminator == 'FPSE':
            from network.Discriminator import FPSEDiscriminator
            self.net['D'] = FPSEDiscriminator(self.args)
            
        for net in self.net.keys():
            self.net[net] = self.net[net].cuda()
            if net == 'P':
                pass
            else:
                self.net[net] = network.warp_network_in_dataparallel(self.net[net], self.args.local_rank)

    def set_optimizer(self):
        self.optims = dict()
        self.schedulers = dict()
        for net in self.net.keys():
            if net == 'T':
                self.optims['T'], self.schedulers['T'] = optimizer.get_optimizer(self.args, self.net['T'])
            elif net == 'D':
                if self.args.train_mode == 'Domain_transfer':
                    self.optims['D'] = optimizer.get_optimizer_MTDT(self.net['D'])
                else:
                    self.optims['D'] = optimizer.get_optimizer_D(self.args, self.net['D'])

            elif net == 'DT':
                self.optims['DT'] = optimizer.get_optimizer_MTDT(self.net['DT'])

    def load_networks(self):
        if self.args.snapshot:
            self.epoch, mean_iu = optimizer.load_weights(self.net['T'], self.optims['T'], self.schedulers['T'],
                            self.args.snapshot, self.args.restore_optimizer)
            if self.args.restore_optimizer is True:
                if self.args.train_mode != 'Soucre_only':
                    self.iter_per_epoch = len(self.target_train_loader)
                else:
                    self.iter_per_epoch = len(self.source_train_loader)
                self.i = self.iter_per_epoch * self.epoch
            else:
                self.epoch = 0

        if self.args.DT_snapshot:
            checkpoint = torch.load(self.args.DT_snapshot, map_location=torch.device('cpu'))
            self.net['DT'].load_state_dict(checkpoint)

    def create_ema_model(self):
        self.ema_model = copy.deepcopy(self.net['T'])
        for param in self.ema_model.module.parameters():
            param.detach_()
    
    def update_ema_model(self, alpha=0.99):
        alpha_teacher = alpha
        # alpha_teacher = min(1 - 1 / (self.i + 1), alpha)
        for ema_param, param in zip(self.ema_model.module.parameters(), self.net['T'].module.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def set_train(self):
        for net in self.net.values():
            net.train()

    def set_zero_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def get_batch(self, tgt=False):

        try:
            _, data = next(self.src_iter)
        except StopIteration:
            self.src_iter = enumerate(self.source_train_loader)
            _, data = next(self.src_iter)

        self.img_src, self.gt_src = data[0], data[1]
        C, H, W = self.img_src.shape[-3:]

        self.img_src = self.img_src.view(-1,C,H,W).cuda()
        self.gt_src = self.gt_src.view(-1,H,W).cuda()
        # self.img_src, self.gt_src = self.img_src.cuda(), self.gt_src.cuda()

        self.batch_pixel_size = C * H * W

        if tgt:
            try:
                _, data = next(self.tgt_iter)
            except StopIteration:
                self.tgt_iter = enumerate(self.target_train_loader)
                _, data = next(self.tgt_iter)
            
            self.img_tgt = data[0]
            self.img_tgt = self.img_tgt.view(-1,C,H,W).cuda()
            # self.img_tgt = self.img_tgt.cuda()
        else:
            self.img_tgt = None

    def calculate_loss(self, loss):
        log_loss = loss.clone().detach_()
        torch.distributed.all_reduce(log_loss, torch.distributed.ReduceOp.SUM)
        log_loss = log_loss / self.args.world_size
        if self.args.insufficient_gpu_memory:
            log_loss = log_loss * 2
        return log_loss
        
    def train_MTDT(self):
        input_imgs = dict()
        input_imgs[self.args.source_dataset[0]] = self.img_src
        for t, target in enumerate(self.args.target_dataset):
            input_imgs[target] = self.img_tgt[t].unsqueeze(0)
        
        self.net['DT'].module.update(input_imgs)
        
        # update D
        loss_dis = self.net['DT'](input_imgs, self.gt_src, netD=self.net['D'], mode='dis')
        self.losses['dis'].update(self.calculate_loss(loss_dis), self.batch_pixel_size)
        loss_dis.backward()
        self.optims['D'].step()

        # update MTDTNet
        self.set_zero_grad()
        loss_recon, loss_adv, loss_per = self.net['DT'](input_imgs, self.gt_src, 
                                    netD=self.net['D'], netP=self.net['P'], mode='gen')
        self.losses['recon'].update(self.calculate_loss(loss_recon), self.batch_pixel_size)
        self.losses['adv'].update(self.calculate_loss(loss_adv), self.batch_pixel_size)
        self.losses['per'].update(self.calculate_loss(loss_per), self.batch_pixel_size)

        loss_MTDT = loss_recon + loss_adv + loss_per
        loss_MTDT.backward()
        self.optims['DT'].step()

    def source_only(self):
        loss = self.net['T'](self.img_src, lbl_src=self.gt_src)
        total_loss = loss['src']
        self.losses['src'].update(self.calculate_loss(total_loss), self.batch_pixel_size)

        total_loss.backward()
        self.optims['T'].step()
        self.schedulers['T'].step()
        
    def warm_up(self):
        if 'DT' in self.net.keys():
            with torch.no_grad():
                transferred_imgs = self.net['DT'](self.img_src, self.gt_src, netD=None, mode='transfer')
                transferred_imgs = torch.cat([transferred_imgs[target] for target in self.args.target_dataset], dim=0)
                input_src = torch.cat([self.img_src, transferred_imgs], dim=0)
                lbl_src = torch.cat([self.gt_src for tgt in range(len(self.args.target_dataset)+len(self.args.source_dataset))], dim=0)
        else:
            input_src = self.img_src
            lbl_src = self.gt_src
        if 'D' not in self.net.keys():
            loss = self.net['T'](input_src, lbl_src=lbl_src)
            total_loss = loss['src']
            self.losses['src'].update(self.calculate_loss(total_loss), self.batch_pixel_size)

            total_loss.backward()
            self.optims['T'].step()
            self.schedulers['T'].step()
        else:
            # update T
            loss, src_out, tgt_out = self.net['T'](src=input_src, lbl_src=lbl_src, tgt=self.img_tgt, netD=self.net['D'])
            loss_src, loss_tgt = loss['src'], loss['tgt']
            self.losses['src'].update(self.calculate_loss(loss_src), self.batch_pixel_size)
            self.losses['tgt'].update(self.calculate_loss(loss_tgt), self.batch_pixel_size)
            total_loss = loss_src+loss_tgt
            total_loss.backward()
            self.optims['T'].step()
            self.schedulers['T'].step()

            self.set_zero_grad()
            # update D
            loss_D = self.net['D'](tgt_out.detach(), x2=src_out.detach(), img=self.img_tgt, img2=input_src, mode='dis')
            self.losses['dis'].update(self.calculate_loss(loss_tgt), self.batch_pixel_size)
            loss_D.backward()
            self.optims['D'].step()

    def adaptation(self):
        lbl_src = self.gt_src.clone().detach()
        input_src = self.img_src.clone().detach()
        input_tgt, lbl_tgt = None, None
        quality_src, quality_tgt = None, None
        if 'DT' in self.net.keys():
            with torch.no_grad():
                transferred_imgs = self.net['DT'](self.img_src, self.gt_src, netD=None, mode='transfer')
                transferred_imgs = torch.cat([transferred_imgs[target] for target in self.args.target_dataset], dim=0)
                input_src = torch.cat([input_src, transferred_imgs], dim=0)
                lbl_trs = torch.cat([self.gt_src for tgt in range(len(self.args.target_dataset))], dim=0)
        else:
            transferred_imgs = None
        # pseudo_label
        if self.args.self_training is not None:
            input_tgt = self.img_tgt.clone().detach()     
            if self.args.self_training == 'bars':
                lbl_trs, lbl_tgt, quality_src, quality_tgt = self.bars(self.ema_model, transferred_imgs, 
                                                                       self.img_tgt, lbl_trs, epoch=self.epoch)
                # lbl_src = torch.cat([lbl_src, lbl_trs], dim=0)
            else:
                lbl_tgt = self.ema_model.module.get_pseudo_label(self.img_tgt)
        
            if self.args.mix_mode is not None:
                input_tgt, lbl_tgt = self.mixer(self.img_src, self.img_tgt, self.gt_src, lbl_tgt, trs=transferred_imgs)
                self.img_mix = input_tgt.clone().detach()
            self.lbl_tgt = lbl_tgt.clone().detach()

        if 'DT' in self.net.keys():
            lbl_src = torch.cat([lbl_src, lbl_trs], dim=0)

        loss = self.net['T'](src=input_src, lbl_src=lbl_src, tgt=input_tgt, lbl_tgt=lbl_tgt,
                            quality_src=quality_src, quality_tgt=quality_tgt, rce=self.args.rce)
        loss_src= loss['src']
        if self.args.self_training is not None:
            loss_tgt = loss['tgt']
        else:
            loss_tgt = torch.zeros_like(loss_src)

        self.losses['src'].update(self.calculate_loss(loss_src), self.batch_pixel_size)
        self.losses['tgt'].update(self.calculate_loss(loss_tgt), self.batch_pixel_size)
        total_loss = loss_src+loss_tgt
        total_loss.backward()
        self.optims['T'].step()
        self.schedulers['T'].step()

    def log_msg(self):
        loss_msg = ''
        for key, value in self.losses.items():
            loss_msg += ('[{} {:0.2f} ],'.format(key, value.avg))

        msg = '[epoch {}], [iter {} / {} : {}], {} [time {:0.4f}]'.format(self.epoch, self.i % self.iter_per_epoch, 
                    self.iter_per_epoch, self.i , loss_msg, self.time_meter.avg / self.args.train_batch_size)
        
        logging.info(msg)

    def train(self):
        
        self.set_train()
        torch.cuda.empty_cache()

        self.src_iter = enumerate(self.source_train_loader)
        self.tgt_iter = enumerate(self.target_train_loader)

        if self.args.self_training == 'bars':
            self.bars.init_centroid(self.source_train_loader, self.target_train_loader, 
                                        self.ema_model, self.net['DT'])
        
        while self.i < self.args.max_iter:
            # Update EPOCH CTR
            cfg.immutable(False)
            cfg.ITER = self.i
            cfg.immutable(True)

            # if self.i % self.iter_per_epoch == 1:
                # if self.args.self_training == 'bars':
                #     self.bars.init_centroid(self.source_train_loader, self.target_train_loader, 
                #                                 self.ema_model, self.net['DT'])

            self.get_batch(tgt = (self.args.train_mode != 'Source_only'))

            self.set_train()
            self.set_zero_grad()

            start_ts = time.time()

            if self.args.train_mode == 'Domain_transfer':
                self.train_MTDT()
            elif self.args.train_mode == 'Source_only':
                self.source_only()
            elif self.args.train_mode == 'Warm_up':
                self.warm_up()
            elif self.args.train_mode == 'Adaptation':
                self.adaptation()
            else:
                print("Wrong train_mode!")

            self.time_meter.update(time.time() - start_ts)

            if self.args.ema:
                if self.i % self.args.ema_update_iter == 0:
                    self.update_ema_model()

            if self.args.local_rank == 0:
                if (self.i % self.iter_per_epoch) % 500 == 1:
                    self.log_msg()
                    if self.args.tensorboard_visualize:
                        self.tensorboard_visualize()
                    if self.args.train_mode == 'Domain_transfer':
                        torch.save(self.net['DT'].state_dict(), self.args.exp_path + '/MTDT_%d.pth'%self.i)
                        print("MTDTNet model saved: %s"%self.args.exp_path)

            # shuffle data each epoch
            if self.i % self.iter_per_epoch == 0:
                if self.args.train_mode != 'Domain_transfer':
                    self.validate_all()  # validate & save network each epoch
                self.epoch += 1
                self.source_train_loader.sampler.set_epoch(self.epoch)
                self.target_train_loader.sampler.set_epoch(self.epoch)

                if self.args.class_uniform_pct:
                    if self.epoch >= self.args.max_cu_epoch:
                        self.source_train_obj.build_epoch(cut=True)
                        self.source_train_loader.sampler.set_num_samples()
                    else:
                        self.source_train_obj.build_epoch()
    
            self.i += 1
        if self.args.train_mode != 'Domain_transfer':
            self.validate_all()  # validate & save network after training

    def tensorboard_visualize(self):
        self.visualizer.visualize_imgs(self.img_src, self.i, title='1_img_src')
        self.visualizer.visualize_imgs(self.img_tgt, self.i, title='2_img_tgt')
        self.visualizer.visualize_labels(self.gt_src, self.i, title='1_gt_src')

        if self.args.train_mode == 'Domain_transfer':
            with torch.no_grad():
                transferred_imgs = self.net['DT'](self.img_src, self.gt_src, netD=None, mode='transfer')
                transferred_imgs = torch.cat([transferred_imgs[target] for target in self.args.target_dataset], dim=0)
            self.visualizer.visualize_imgs(transferred_imgs, self.i, title='3_img_trs')
        else:
            transferred_imgs = None
        if self.args.train_mode == 'Warm_up' or self.args.train_mode == 'Adaptation':
            self.visualizer.visualize_labels(self.lbl_tgt, self.i, title='2_lbl_tgt')
        if self.args.mix_mode is not None:
            self.visualizer.visualize_imgs(self.img_mix, self.i, title='4_img_mix')

        # sink
        self.visualizer.visualize_imgs(self.img_src, self.i, title='1_img_src')

    def validate(self, val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, best_miou, save_pth=True):
        """
        Runs the validation loop after each training epoch
        val_loader: Data loader for validation
        dataset: dataset name (str)
        net: thet network
        criterion: loss fn
        optimizer: optimizer
        curr_epoch: current epoch
        writer: tensorboard writer
        return: val_avg for step function if required
        """

        net.eval()
        val_loss = AverageMeter()
        iou_acc = 0
        error_acc = 0
        dump_images = []

        for val_idx, data in enumerate(val_loader):
            # input        = torch.Size([1, 3, 713, 713])
            # gt_image           = torch.Size([1, 713, 713])
            inputs, gt_image, img_names, _ = data

            if len(inputs.shape) == 5:
                B, D, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                gt_image = gt_image.view(-1, 1, H, W)
            # print(inputs.size(), gt_image.size())

            assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
            assert inputs.size()[2:] == gt_image.size()[1:]

            batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
            inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

            with torch.no_grad():
                output = net(inputs)
                # output = output[0]

            del inputs

            assert output.size()[2:] == gt_image.size()[1:]
            assert output.size()[1] == self.args.num_classes

            val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

            del gt_cuda

            # Collect data from different GPU to a single GPU since
            # encoding.parallel.criterionparallel function calculates distributed loss
            # functions
            predictions = output.data.max(1)[1].cpu()

            # Logging
            if val_idx % 100 == 0:
                if self.args.local_rank == 0:
                    logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
            if val_idx > 10 and self.args.test_mode:
                break

            # Image Dumps
            if val_idx < 10:
                dump_images.append([gt_image, predictions, img_names])

            iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                                self.args.num_classes)
            del output, val_idx, data

        iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
        torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
        iou_acc = iou_acc_tensor.cpu().numpy()

        if self.args.local_rank == 0:
            evaluate_eval(self.args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                        writer, best_miou, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

        return val_loss.avg

    def validate_all(self):
        # The others: evaluation of domain adaptation
        for dataset, val_loader in self.extra_val_loaders.items():
            print("validating...")
            # print("Extra validating... This won't save pth file")
            self.validate(val_loader, dataset, self.net['T'], self.criterion_val, self.optims['T'], self.schedulers['T'], 
                            self.epoch, self.writer, self.i, self.best_miou, save_pth=(dataset == self.args.val_dataset[-1]))
        print(self.best_miou)
        self.set_train()

