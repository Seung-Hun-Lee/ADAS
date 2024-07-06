# modified from https://github.com/microsoft/ProDA/blob/9ba80c7dbbd23ba1a126e3f4003a72f27d121a1f/models/deeplabv2.py
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import datasets

affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = BatchNorm(planes, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes, affine=affine_par)
        # for i in self.bn2.parameters():
        #     i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4, affine=affine_par)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes//r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes//r, inplanes),
                nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)

class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate = 0.1, use_se = True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
                nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))

        for dilation, padding in zip(dilation_series, padding_series):
            #self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True), 
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))
 
        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                        nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                        nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out

class ResNet101(nn.Module):
    def __init__(self, args, block, layers, num_classes, BatchNorm, bn_clr=False, criterion=None):
        self.args = args
        self.inplanes = 64
        self.bn_clr = bn_clr
        super(ResNet101, self).__init__()
        self.criterion = criterion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, BatchNorm=BatchNorm)
        #self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer5 = self._make_pred_layer(Classifier_Module2, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.embed = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
        )
        if self.bn_clr:
            self.bn_pretrain = BatchNorm(2048, affine=affine_par)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion, affine=affine_par))
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, src=None, lbl_src=None, tgt=None, lbl_tgt=None, rce=True, reg=True,
                quality_src=None, quality_tgt=None, netD=None, eval_mode='default'):
        
        if src is not None:
            _, _, h, w = src.size()
            out = self.infer(src)
            x = nn.functional.interpolate(out['out'], size=(h, w), mode='bilinear', align_corners=True)

        if tgt is not None:
            _, _, h, w = tgt.size()
            out2 = self.infer(tgt)
            x2 = nn.functional.interpolate(out2['out'], size=(h, w), mode='bilinear', align_corners=True)
            
        if self.training:
            losses = dict()
            if lbl_src is not None:
                loss_src = self.criterion(x, lbl_src, weight=quality_src)
                losses['src'] = loss_src
            if netD is not None:
                    losses['tgt'] = 0.01 * netD(x2, img=tgt, mode='gen')
                    return losses, x, x2
            if lbl_tgt is not None:
                loss_tgt = 0.1 * self.criterion(x2,lbl_tgt.clone().detach(), weight=quality_tgt)
                if rce: 
                    loss_tgt = loss_tgt + 1.0 * self.rce(x2, lbl_tgt)  # lbl_tgt: Pseudo Label
                # if reg:
                #     loss_tgt = loss_tgt + 0.1 * self.reg(out2['out'])
                losses['tgt'] = loss_tgt
            return losses
        else:
            if eval_mode == 'default':
                if tgt is not None:
                    return x, x2
                else:
                    return x
            elif eval_mode == 'feat':
                if tgt is not None:
                    return out, out2
                else:
                    return out
            elif eval_mode == 'all':
                if tgt is not None:
                    return x, x2, out, out2
                else:
                    return x, out
                
    def rce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        mask = (labels != 255).float()
        labels[labels==255] = self.args.num_classes
        label_one_hot = torch.nn.functional.one_hot(labels, self.args.num_classes + 1).float().cuda()
        label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0)
        rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
        return rce

    def reg(self, p):
        logp = F.log_softmax(p, dim=1)
        return - logp.sum() / (logp.shape[0]*logp.shape[1]*logp.shape[2]*logp.shape[3])
    
    def get_pseudo_label(self, img, thresh=0.968):
        with torch.no_grad():
            _, _, h, w = img.size()
            out = self.infer(img)
            x = nn.functional.interpolate(out['out'], size=(h, w), mode='bilinear', align_corners=True)
            x = F.softmax(x, dim=1)
            max_pred, out = x.max(dim=1, keepdim=False)
            out[max_pred<thresh] = datasets.ignore_label
        return out.long()
    
    def infer(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.bn_clr:
            x = self.bn_pretrain(x)
        out = self.layer5(x, get_feat=True)
        return out
    
    def get_1x_lr_params(self):

        b = []
        
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        if self.bn_clr:
            b.append(self.bn_pretrain.parameters())    
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]
    
    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10  
            
    def CrossEntropy2d(self, input, target, weight=None, size_average=True, softmax_used=False, reduction='mean', cls_num_list=None):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
            raise NotImplementedError('sizes of input and label are not consistent')

        # input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        # target = target.view(-1)
        if softmax_used:
            loss = F.nll_loss(
                input, target, weight=weight, size_average=size_average, ignore_index=-1
            )
        else:
            loss = F.cross_entropy(
                input, target, weight=weight, size_average=size_average, ignore_index=-1, reduction=reduction
            )
        return loss  

def freeze_bn_func(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def Deeplab(args, BatchNorm=nn.BatchNorm2d, num_classes=21, freeze_bn=False, restore_from=None, initialization=None, bn_clr=False, criterion=None):
    model = ResNet101(args, Bottleneck, [3, 4, 23, 3], num_classes, BatchNorm, bn_clr=bn_clr, criterion=criterion)
    if freeze_bn:
        model.apply(freeze_bn_func)
    if initialization is None:
        pass
    else:
        if initialization == 'imagenet':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        else:
            pretrain_dict = torch.load(initialization)
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                # print(k)
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    if restore_from is not None: 
        model.load_state_dict(torch.load(restore_from, map_location=lambda storage, loc: storage))
    
    return model