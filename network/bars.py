import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
import torch.nn.functional as F
import torch.nn as nn
import datasets


class Bidirectional_Adaptive_Region_Selection(nn.Module):
    def __init__(self, args):
        super(Bidirectional_Adaptive_Region_Selection, self).__init__()
        self.args = args
        self.num_classes = self.args.num_classes
        self.feat_ch = self.args.feat_ch

        self.Centroid_tgt = nn.ParameterDict()
        self.Amount_tgt = nn.ParameterDict()
        self.Centroid_trs = nn.ParameterDict()
        self.Amount_trs = nn.ParameterDict()

        for tgt in self.args.target_dataset:
            self.Centroid_tgt[tgt] = nn.Parameter(torch.zeros(self.num_classes, self.feat_ch)
                                                  .cuda(non_blocking=True), requires_grad=False)
            self.Amount_tgt[tgt] = nn.Parameter(torch.zeros(self.num_classes)
                                                  .cuda(non_blocking=True), requires_grad=False)
            self.Centroid_trs[tgt] = nn.Parameter(torch.zeros(self.num_classes, self.feat_ch)
                                                  .cuda(non_blocking=True), requires_grad=False)
            self.Amount_trs[tgt] = nn.Parameter(torch.zeros(self.num_classes)
                                                  .cuda(non_blocking=True), requires_grad=False)
        
    def init_centroid(self, source_train_loader, target_train_loader, netT, netDT):
        print('initialize centroids... (This will be done within a few minutes)')
        netT.eval()
        source_iter = enumerate(source_train_loader)
        for i, data in enumerate(target_train_loader):
            if torch.min(torch.cat(list(self.Amount_tgt.values()),dim=0)) > 10 \
            and torch.min(torch.cat(list(self.Amount_trs.values()),dim=0)) > 10:
                break
            
            input_tgt, *_ = data
            try:
                _, input_src = next(source_iter)
            except StopIteration:
                source_iter = enumerate(source_train_loader)
                _, input_src = next(source_iter)
            input_src, gt_src = input_src[0], input_src[1]

            C, H, W = input_src.shape[-3:]

            input_src = input_src.view(-1,C,H,W)
            gt_src = gt_src.view(-1,H,W)
            input_tgt = input_tgt.view(-1,C,H,W)
            input_src, gt_src = input_src.cuda(), gt_src.cuda()
            input_tgt = input_tgt.cuda()

            with torch.no_grad():
                transferred_imgs = netDT(input_src, gt_src, netD=None, mode='transfer')
                transferred_imgs = torch.cat([transferred_imgs[target] for target in self.args.target_dataset], dim=0)
                gt_trs = torch.cat([gt_src for tgt in range(len(self.args.target_dataset))], dim=0)
                # gt_src = torch.cat([gt_src, gt_trs], dim=0)
                # input_src = torch.cat([input_src, transferred_imgs], dim=0)
            self.update_all_centroids(transferred_imgs, input_tgt, gt_trs, netT)
        print('done')

    def update_all_centroids(self, img_trs, img_tgt, gt_trs, netT, adaptive_label_tgt=None, thresh=0.95):
        with torch.no_grad():
            out_trs, out_tgt = netT(src=img_trs, tgt=img_tgt, eval_mode='feat')
            feat_trs, pred_trs = out_trs['feat'], out_trs['out']
            feat_tgt, pred_tgt = out_tgt['feat'], out_tgt['out']
            B, C, H, W = feat_trs.size()

            if adaptive_label_tgt is not None:
                pred_tgt = adaptive_label_tgt
                pred_tgt = F.interpolate(pred_tgt.unsqueeze(0).float(), size=(H, W), mode='nearest').squeeze(0).long()
            else:
                max_tgt, pred_tgt = F.softmax(pred_tgt, dim=1).max(dim=1, keepdim=False)
                pred_tgt[max_tgt<thresh] = datasets.ignore_label
            
            pred_trs = pred_trs.argmax(dim=1)
            mask = F.interpolate(gt_trs.unsqueeze(0).float(), size=(H, W), mode='nearest').squeeze(0).long()
            pred_trs[pred_trs!=mask] = datasets.ignore_label
            
            feat_trs = feat_trs.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            feat_tgt = feat_tgt.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            pred_trs = pred_trs.contiguous().view(B * H * W, )
            pred_tgt = pred_tgt.contiguous().view(B * H * W, )

            feat_trs = feat_trs.chunk(len(self.args.target_dataset), dim=0)
            feat_tgt = feat_tgt.chunk(len(self.args.target_dataset), dim=0)
            pred_trs = pred_trs.chunk(len(self.args.target_dataset), dim=0)
            pred_tgt = pred_tgt.chunk(len(self.args.target_dataset), dim=0)

            for (t, tgt) in enumerate(self.args.target_dataset):
                mean_trs, sum_weight_trs, class_dist_trs = self.gather_centroids(feat_trs[t], pred_trs[t])
                mean_tgt, sum_weight_tgt, class_dist_tgt = self.gather_centroids(feat_tgt[t], pred_tgt[t])
                self.update(mean_trs, mean_tgt, sum_weight_trs, sum_weight_tgt, class_dist_trs, class_dist_tgt, tgt)
    
    def gather_centroids(self, feat, pred):
        mean, sum_weight, class_dist = self.get_centroid_vector(feat, pred, ignore_label=datasets.ignore_label)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            mean_all = [torch.ones_like(mean) for _ in range(self.args.world_size)]
            sum_weight_all = [torch.ones_like(sum_weight) for _ in range(self.args.world_size)]
            class_dist_all = [torch.ones_like(class_dist) for _ in range(self.args.world_size)]
            torch.distributed.all_gather(mean_all, mean.clone())
            torch.distributed.all_gather(sum_weight_all, sum_weight.clone())
            torch.distributed.all_gather(class_dist_all, class_dist.clone())
        else:
            mean_all, sum_weight_all, class_dist_all = mean, sum_weight, class_dist
        return mean_all, sum_weight_all, class_dist_all
        
    def get_centroid_vector(self, features, labels, ignore_label=datasets.ignore_label):
        mask = (labels != ignore_label)
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]
        N, A = features.size()
        C = self.num_classes
        # refer to SDCA for fast implementation
        features = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = features.mul(NxCxA_onehot)
        Amount_CXA = NxCxA_onehot.sum(0)
        Amount_CXA[Amount_CXA == 0] = 1
        mean = features_by_sort.sum(0) / Amount_CXA
        sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
        class_dist = onehot.sum(0)
        return mean, sum_weight, class_dist

    def update_centroid(self, mean_trs, mean_tgt, sum_weight_trs, sum_weight_tgt, class_dist_trs, class_dist_tgt, target):
        C = self.num_classes
        A = self.feat_ch
        weight_trs = sum_weight_trs.div(sum_weight_trs + self.Amount_trs[target].view(C, 1).expand(C, A))
        weight_trs[sum_weight_trs == 0] = 0

        weight_tgt = sum_weight_tgt.div(sum_weight_tgt + self.Amount_tgt[target].view(C, 1).expand(C, A))
        weight_tgt[sum_weight_tgt == 0] = 0

        self.Centroid_trs[target] = nn.Parameter(self.Centroid_trs[target].mul(1 - weight_trs) + mean_trs.mul(weight_trs))
        self.Amount_trs[target] += class_dist_trs

        self.Centroid_tgt[target] = nn.Parameter(self.Centroid_tgt[target].mul(1 - weight_tgt) + mean_tgt.mul(weight_tgt))
        self.Amount_tgt[target] += class_dist_tgt

    def update(self, mean_trs, mean_tgt, sum_weight_trs, sum_weight_tgt, class_dist_trs, class_dist_tgt, target):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for w in range(self.args.world_size):
                self.update_centroid(mean_trs[w], mean_tgt[w], sum_weight_trs[w], sum_weight_tgt[w], class_dist_trs[w], class_dist_tgt[w], target)
        else:
            self.update_centroid(mean_trs, mean_tgt, sum_weight_trs, sum_weight_tgt, class_dist_trs, class_dist_tgt, target)

    def forward(self, net_pseudo, input_trs, input_tgt, gt_trs, epoch=None, threshold=0.968):
        with torch.no_grad():
            out_trs, out_tgt = net_pseudo(src=input_trs, tgt=input_tgt, eval_mode='feat')

            prob_trs = F.interpolate(out_trs['out'], size=gt_trs.shape[1:], mode='bilinear', align_corners=True)
            prob_tgt = F.interpolate(out_tgt['out'], size=gt_trs.shape[1:], mode='bilinear', align_corners=True)
            prob_trs, prob_tgt = F.softmax(prob_trs, dim=1), F.softmax(prob_tgt, dim=1)

            max_trs, pred_trs = prob_trs.max(dim=1, keepdim=True)
            max_tgt, pred_tgt = prob_tgt.max(dim=1, keepdim=True)

            feat_trs, feat_tgt = out_trs['feat'], out_tgt['feat']
            feat_trs = feat_trs.unsqueeze(1).expand(-1, self.num_classes, -1, -1, -1)  # (B x T) x C x N x H' x W'
            feat_tgt = feat_tgt.unsqueeze(1).expand(-1, self.num_classes, -1, -1, -1)

            centroid_trs = torch.cat([self.Centroid_trs[tgt].unsqueeze(0) for tgt in self.args.target_dataset], dim=0) # T x C x N
            centroid_tgt = torch.cat([self.Centroid_tgt[tgt].unsqueeze(0) for tgt in self.args.target_dataset], dim=0)

            centroid_trs = centroid_trs.unsqueeze(-1).unsqueeze(-1).expand_as(feat_trs[:len(self.args.target_dataset)]).repeat(self.args.bs_mult, 1, 1, 1, 1)
            centroid_tgt = centroid_tgt.unsqueeze(-1).unsqueeze(-1).expand_as(feat_tgt[:len(self.args.target_dataset)]).repeat(self.args.bs_mult, 1, 1, 1, 1)

            distance_trs_tgt = torch.norm(feat_trs-centroid_tgt, p=2, dim=2)
            distance_tgt_trs = torch.norm(feat_tgt-centroid_trs, p=2, dim=2)

            distance_trs_tgt = F.interpolate(distance_trs_tgt, size=gt_trs.shape[1:], mode='bilinear', align_corners=True)
            distance_tgt_trs = F.interpolate(distance_tgt_trs, size=gt_trs.shape[1:], mode='bilinear', align_corners=True)

            nearest_classes_trs = torch.argmin(distance_trs_tgt, dim=1).long()  # (B x T) x H' x W'
            nearest_classes_tgt = torch.argmin(distance_tgt_trs, dim=1).long()

            pred_trs, pred_tgt = pred_trs.long(), pred_tgt.long()

            dis_tgt = F.softmax(-distance_tgt_trs, dim=1)
            dis_tgt = dis_tgt.gather(dim=1, index=pred_tgt).squeeze(1)

            dis_trs = F.softmax(-distance_trs_tgt, dim=1)
            dis_trs = dis_trs.gather(dim=1, index=pred_trs).squeeze(1)

            pred_trs, pred_tgt = pred_trs.squeeze(1), pred_tgt.squeeze(1)

            adaptive_label_tgt = pred_tgt.clone()  # B x H x W
            adaptive_label_tgt[nearest_classes_tgt!=adaptive_label_tgt] = datasets.ignore_label
            # adaptive_label_tgt[max_tgt.squeeze(1)<threshold] = datasets.ignore_label
            adaptive_label_tgt[dis_tgt<threshold] = datasets.ignore_label

            adaptive_label_trs = gt_trs.clone()
            pred_trs[nearest_classes_trs!=pred_trs] = datasets.ignore_label
            # adaptive_label_trs[pred_trs!=adaptive_label_trs] = datasets.ignore_label
            adaptive_label_trs[dis_trs<threshold] = datasets.ignore_label

            if self.args.curriculum:
                hard_ratio = (self.args.initial_ratio + epoch * self.args.incremental_ratio)
                if hard_ratio > 1:
                    hard_ratio = 1.0

                ignored_pixels_trs = ((adaptive_label_trs==datasets.ignore_label) & (gt_trs!=datasets.ignore_label)).sum()
                topk_hard_trs = int(ignored_pixels_trs * hard_ratio)  # 10 * (epoch + 1) %
                
                ignored_pixels_tgt = (adaptive_label_tgt==datasets.ignore_label).sum()
                topk_hard_tgt = int(ignored_pixels_tgt * hard_ratio)

                if topk_hard_trs > 1:
                    ignored_dis_trs = torch.masked_select(dis_trs, adaptive_label_trs==datasets.ignore_label)
                    ignored_dis_trs, _ = ignored_dis_trs.sort(descending=True)
                    hard_thresh_trs = ignored_dis_trs[topk_hard_trs-1]
                else:
                    hard_thresh_trs = 0.95

                if topk_hard_tgt > 1:
                    ignored_dis_tgt = torch.masked_select(dis_tgt, adaptive_label_tgt==datasets.ignore_label)
                    ignored_dis_tgt, _ = ignored_dis_tgt.sort(descending=True)
                    hard_thresh_tgt = ignored_dis_tgt[topk_hard_tgt-1]
                else:
                    hard_thresh_tgt = 0.95

                quality_trs = dis_trs
                quality_tgt = dis_tgt

                quality_trs[(adaptive_label_trs == datasets.ignore_label) & (quality_trs<hard_thresh_trs)] = 0
                quality_tgt[(adaptive_label_tgt == datasets.ignore_label) & (quality_tgt<hard_thresh_tgt)] = 0

                topk_adaptive_label_trs = gt_trs.clone()
                topk_adaptive_label_tgt = pred_tgt.clone()

                topk_adaptive_label_trs[quality_trs == 0] = datasets.ignore_label
                topk_adaptive_label_tgt[quality_tgt == 0] = datasets.ignore_label

                adaptive_label_trs, adaptive_label_tgt = topk_adaptive_label_trs, topk_adaptive_label_tgt

                quality_trs = torch.cat([torch.ones_like(quality_trs[:self.args.bs_mult]), quality_trs], dim=0)  # train full source image (Batch data contains [src, trs1, trs2, trs3])
            else:
                quality_trs, quality_tgt = None, None

        self.update_all_centroids(input_trs, input_tgt, adaptive_label_trs, net_pseudo, adaptive_label_tgt=adaptive_label_tgt)

        return adaptive_label_trs, adaptive_label_tgt, quality_trs, quality_tgt
