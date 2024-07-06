import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
import torch.nn.functional as F
import torch.nn as nn
import datasets
import random
import numpy as np


joint_classes = {
    # pole <-> traffic light & sign
    5: [6, 7],
    6: [5],
    7: [5],
    # rider <-> motorcycle & bike
    12: [17, 18],
    17: [12],
    18: [12]
}

class Domain_Mixer(nn.Module):
    '''
    Implementation for single source and multiple targets
    src: source image (B x 1) x 3 x H x W
    tgt: target image (B x T) x H x W
    trs: transferred image (B x 1 x T) x 3 x H x W    
    mode: ['uniform' / 'random' / 'dist'] 'uniform': mix same classes to each target domain. 'random': mix random classes to each target domain.
    '''
    def __init__(self, args):
        super(Domain_Mixer, self).__init__()
        self.args = args
        self.targets = args.target_dataset
        self.num_classes = args.num_classes

        if self.args.mix_mode == 'class_dist' or self.args.mix_mode == 'class_dist_relation' or self.args.mix_mode == 'random+':
            self.class_dist = nn.ParameterDict()

            for tgt in self.targets:
                self.class_dist[tgt] = nn.Parameter(torch.zeros(self.num_classes).cuda(non_blocking=True), requires_grad=False)
    
    def forward(self, img_src, img_tgt, lbl_src, lbl_tgt, epoch=0, trs=None):
        with torch.no_grad():
            if self.args.mix_mode == 'class_dist' or self.args.mix_mode == 'class_dist_relation' or self.args.mix_mode == 'random+':
                for (t, tgt) in enumerate(self.targets):
                    self.update(lbl_tgt[t], tgt)
            mix_imgs = []
            mix_lbls = []

            tgt_imgs = img_tgt.chunk(len(self.args.target_dataset))
            tgt_pred_ = lbl_tgt.chunk(len(self.args.target_dataset))
            if trs is not None:
                trs_imgs = trs.chunk(len(self.args.target_dataset))

            for src_idx in range(img_src.size(0)):
                if self.args.mix_mode == 'uniform':  # same mask for all target domains
                    mask = self.random_select(lbl_src[src_idx], epoch=epoch)

                for (tgt_idx, tgt_img) in enumerate(tgt_imgs):
                    if self.args.mix_mode == 'random':  # different random mask for each target domain
                        mask = self.random_select(lbl_src[src_idx], epoch=epoch)  
                    elif self.args.mix_mode == 'class_dist_relation' or self.args.mix_mode == 'class_dist_relation':
                        mask = self.select_by_class_dist_relation(lbl_src[src_idx], self.targets[tgt_idx])
                    elif self.args.mix_mode == 'random+':
                        mask = self.random_select_dist(lbl_src[src_idx], self.targets[tgt_idx], epoch=epoch)

                    mix_src = img_src[src_idx] if trs is None else trs_imgs[tgt_idx][src_idx]
                    mix_tgt = tgt_img[src_idx]
                    mix_imgs.append(mask.unsqueeze(0) * mix_src + (1-mask.unsqueeze(0))*mix_tgt)
                    mix_lbls.append(mask * lbl_src[src_idx] + (1-mask) * tgt_pred_[tgt_idx][src_idx])
            
            
            mix_imgs = torch.stack(mix_imgs, dim=0)
            mix_lbls = torch.stack(mix_lbls, dim=0)


        return mix_imgs, mix_lbls

        
    def generate_class_mask(self, seg, classes):
        seg, classes = torch.broadcast_tensors(seg.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))

        N = seg.eq(classes).sum(0)

        return N

    def update_tensor(self, N, cut_mix):
        mask = (N == 0) & (cut_mix != 0)
        N[mask] = 1
        return N
        

    def region_selection(self, tensor):
        indices = torch.nonzero(tensor == 1)
        if len(indices) == 0:
            return None

        top = indices[0]
        bottom = indices[0]
        left = indices[0]
        right = indices[0]

        for idx in indices:
            if idx[0] < top[0]:
                top = idx
            if idx[0] > bottom[0]:
                bottom = idx
            if idx[1] < left[1]:
                left = idx
            if idx[1] > right[1]:
                right = idx

        tensor[top[0]:bottom[0]+1, left[1]:right[1]+1] = 1

        return tensor


    def update(self, lbl, target):
        classes = lbl.unique()
        classes = classes[classes!=datasets.ignore_label]
        for cls in classes:
            self.class_dist[target][cls] += 1
            # self.class_dist[target][cls] += (lbl==cls).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.class_dist[target], torch.distributed.ReduceOp.SUM)

    def select_by_class_dist_relation(self, lbl, target):
        selected_classes_idx = []
        source_classes = lbl.unique()
        source_classes = source_classes[source_classes!=datasets.ignore_label]
        num_select= source_classes.size(0) // 2

        # class_dist
        target_class_dist = self.class_dist[target].clone()
        _, target_class_ascending_order = target_class_dist.sort(descending=False)

        for cls in target_class_ascending_order:
            if len(selected_classes_idx) >= num_select:
                break
            if cls in source_classes:
                cls_idx = (source_classes==cls).nonzero(as_tuple=True)[0].item()
                if cls_idx not in selected_classes_idx:
                    selected_classes_idx.append(cls_idx)
                # relation
                if (self.args.mix_mode == 'class_dist_relation') and (int(cls) in joint_classes.keys()):
                    for joint_cls in joint_classes[int(cls)]:
                        if joint_cls in source_classes and self.is_adjacent(lbl, cls, joint_cls):
                            joint_idx = (source_classes==joint_cls).nonzero(as_tuple=True)[0].item()
                            if joint_idx not in selected_classes_idx:
                                selected_classes_idx.append(joint_idx)
                                
        selected_classes_idx = torch.as_tensor(selected_classes_idx, dtype=torch.long, device=lbl.device)
        selected_classes = source_classes[selected_classes_idx]
        mask = self.generate_class_mask(lbl, selected_classes)
        return mask

    def is_adjacent(self, lbl, cls1, cls2):
        mask = torch.zeros_like(lbl)
        mask[lbl==cls1] = cls1
        mask[lbl==cls2] = cls2
        mask_vertical = mask[1:,:] + mask[:-1,:]
        mask_horizontal = mask[:,1:] + mask[:,:-1]
        if (cls1+cls2 in mask_vertical) or (cls1+cls2 in mask_horizontal):
            return True
        else:
            return False

    def random_select(self, lbl, epoch=0):
        classes = lbl.unique()
        classes = classes[classes!=datasets.ignore_label]
        num_classes = classes.size(0)
        prop = 0.5 - (epoch*0.05)
        num_samples = int(torch.sum(torch.bernoulli(prop*torch.ones(num_classes))))
        selected_classes_idx = torch.randperm(num_classes)[:num_samples]
        selected_classes = classes[selected_classes_idx]
        mask = self.generate_class_mask(lbl, selected_classes)
        return mask

    def random_select_dist(self, lbl, target, epoch=0):
        classes = lbl.unique()
        classes = classes[classes!=datasets.ignore_label]
        num_classes = classes.size(0)
        prop = 0.5 - (epoch*0.05)
        num_samples = int(torch.sum(torch.bernoulli(prop*torch.ones(num_classes))))
        selected_classes_idx = torch.randperm(num_classes)[:num_samples]
        selected_classes = classes[selected_classes_idx]

        # class_dist
        target_class_dist = self.class_dist[target].clone()
        _, target_class_ascending_order = target_class_dist.sort(descending=False)

        # top 5 rare class
        for cls in target_class_ascending_order[:5]:
            if cls in classes and cls not in selected_classes:
                # selected_classes = torch.cat(selected_classes, cls)
                cls = torch.tensor(cls)
                cls = torch.unsqueeze(cls, dim=0)
                selected_classes = torch.cat((selected_classes, cls), dim=0)

                break

        mask = self.generate_class_mask(lbl, selected_classes)

        return mask
    