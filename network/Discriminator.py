# modified from https://github.com/microsoft/ProDA/blob/main/models/discriminator.py


import sys
import torch
import re
from collections import OrderedDict
import os.path
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from network.mynn import initialize_weights
import torch.nn.utils.spectral_norm as spectral_norm


def get_nonspade_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer, args):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)        
            
        if subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        elif subnorm_type == 'sync_batch' and args.mpdist:
            norm_layer = nn.SyncBatchNorm(get_out_channel(layer), affine=True)
        else:
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# AdaptSeg (CVPR'18)
class FCDiscriminator(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, args, planes = 64):
        self.args = args
        inplanes = args.num_classes
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(planes*8, 1, kernel_size=1)
        initialize_weights(self.conv1)
        initialize_weights(self.conv2)
        initialize_weights(self.conv3)
        initialize_weights(self.conv4)
        initialize_weights(self.relu)
        initialize_weights(self.leaky_relu)
        initialize_weights(self.classifier)

    def forward(self, x, x2=None, mode=None, img=None, img2=None):
        '''
        x: fake (target prediction)
        x2: real (source prediction)
        mode: dis or gen
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        if x2 is not None:
            x2 = self.conv1(x2)
            x2 = self.relu(x2)
            x2 = self.conv2(x2)
            x2 = self.relu(x2)
            x2 = self.conv3(x2)
            x2 = self.relu(x2)
            x2 = self.conv4(x2)
            x2 = self.leaky_relu(x2)
            x2 = self.classifier(x2)

        # LSGAN Loss
        if mode == 'dis':
            real_loss = F.mse_loss(x2, torch.zeros_like(x2))
            fake_loss = F.mse_loss(x, torch.ones_like(x))
            loss = real_loss + fake_loss
        elif mode == 'gen':
            loss = F.mse_loss(x, torch.zeros_like(x))

        return loss
    

# FPSE Discriminator (NuerIPS'19)
class FPSEDiscriminator(nn.Module):
    def __init__(self, args):
        super(FPSEDiscriminator, self).__init__()
        self.args = args
        nf = 64
        input_nc = 3
        label_nc = args.num_classes
        
        norm_layer = get_nonspade_norm_layer("spectralinstance")

        # bottom-up pathway
        self.enc1 = nn.Sequential(
                norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1), args), 
                nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(
                norm_layer(nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1), args), 
                nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(
                norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1), args), 
                nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(
                norm_layer(nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1), args), 
                nn.LeakyReLU(0.2, True))
        self.enc5 = nn.Sequential(
                norm_layer(nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=2, padding=1), args), 
                nn.LeakyReLU(0.2, True))

        # top-down pathway
        self.lat2 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1), args), 
                    nn.LeakyReLU(0.2, True))
        self.lat3 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*4, nf*4, kernel_size=1), args), 
                    nn.LeakyReLU(0.2, True))
        self.lat4 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1), args), 
                    nn.LeakyReLU(0.2, True))
        self.lat5 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1), args), 
                    nn.LeakyReLU(0.2, True))
        
        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # final layers
        self.final2 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1), args), 
                    nn.LeakyReLU(0.2, True))
        self.final3 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1), args), 
                    nn.LeakyReLU(0.2, True))
        self.final4 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1), args), 
                    nn.LeakyReLU(0.2, True))
    
        # true/false prediction and semantic alignment prediction
        # self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)
        
        initialize_weights(self.enc1)        
        initialize_weights(self.enc2)        
        initialize_weights(self.enc3)        
        initialize_weights(self.enc4)        
        initialize_weights(self.enc5)
                
        initialize_weights(self.lat2)
        initialize_weights(self.lat3)
        initialize_weights(self.lat4)
        initialize_weights(self.lat5)  
        
        initialize_weights(self.final2)
        initialize_weights(self.final3)
        initialize_weights(self.final4)

        initialize_weights(self.seg)
        initialize_weights(self.embedding)


    def forward(self, x, x2=None, mode=None, img=None, img2=None):
        '''
        x: fake (target prediction)
        x2: real (source prediction)
        mode: dis or gen
        '''
        # bottom-up pathway
        feat11_fake = self.enc1(img)
        feat12_fake = self.enc2(feat11_fake)
        feat13_fake = self.enc3(feat12_fake)
        feat14_fake = self.enc4(feat13_fake)
        feat15_fake = self.enc5(feat14_fake)
        # top-down pathway and lateral connections
        feat25_fake = self.lat5(feat15_fake)
        feat24_fake = self.up(feat25_fake) + self.lat4(feat14_fake)
        feat23_fake = self.up(feat24_fake) + self.lat3(feat13_fake)
        feat22_fake = self.up(feat23_fake) + self.lat2(feat12_fake)
        # final prediction layers
        feat32_fake = self.final2(feat22_fake)
        feat33_fake = self.final3(feat23_fake)
        feat34_fake = self.final4(feat24_fake)
        
        seg2_fake = self.seg(feat32_fake)
        seg3_fake = self.seg(feat33_fake)
        seg4_fake = self.seg(feat34_fake)

        if img2 is not None:
            # bottom-up pathway
            feat11_real = self.enc1(img2)
            feat12_real = self.enc2(feat11_real)
            feat13_real = self.enc3(feat12_real)
            feat14_real = self.enc4(feat13_real)
            feat15_real = self.enc5(feat14_real)
            # top-down pathway and lateral connections
            feat25_real = self.lat5(feat15_real)
            feat24_real = self.up(feat25_real) + self.lat4(feat14_real)
            feat23_real = self.up(feat24_real) + self.lat3(feat13_real)
            feat22_real = self.up(feat23_real) + self.lat2(feat12_real)
            # final prediction layers
            feat32_real = self.final2(feat22_real)
            feat33_real = self.final3(feat23_real)
            feat34_real = self.final4(feat24_real)
            
            seg2_real = self.seg(feat32_real)
            seg3_real = self.seg(feat33_real)
            seg4_real = self.seg(feat34_real)

        # intermediate features for discriminator feature matching loss
        # feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segemb_fake = self.embedding(x)
        segemb_fake = F.avg_pool2d(segemb_fake, kernel_size=2, stride=2)
        segemb2_fake = F.avg_pool2d(segemb_fake, kernel_size=2, stride=2)
        segemb3_fake = F.avg_pool2d(segemb2_fake, kernel_size=2, stride=2)
        segemb4_fake = F.avg_pool2d(segemb3_fake, kernel_size=2, stride=2)
        if x2 is not None:
            segemb_real = self.embedding(x2)
            segemb_real = F.avg_pool2d(segemb_real, kernel_size=2, stride=2)
            segemb2_real = F.avg_pool2d(segemb_real, kernel_size=2, stride=2)
            segemb3_real = F.avg_pool2d(segemb2_real, kernel_size=2, stride=2)
            segemb4_real = F.avg_pool2d(segemb3_real, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        score_fake2 = torch.mul(segemb2_fake, seg2_fake).sum(dim=1, keepdim=True)
        score_fake3 = torch.mul(segemb3_fake, seg3_fake).sum(dim=1, keepdim=True)
        score_fake4 = torch.mul(segemb4_fake, seg4_fake).sum(dim=1, keepdim=True)
        if x2 is not None:
            score_real2 = torch.mul(segemb2_real, seg2_real).sum(dim=1, keepdim=True)
            score_real3 = torch.mul(segemb3_real, seg3_real).sum(dim=1, keepdim=True)
            score_real4 = torch.mul(segemb4_real, seg4_real).sum(dim=1, keepdim=True)

        # adversarial_loss

        if mode=='gen':
            loss = -score_fake2.mean() - score_fake3.mean() - score_fake4.mean()
            return loss
        
        elif mode=='dis':
            fake_loss = F.relu(1. + score_fake2).mean() + F.relu(1. + score_fake3).mean() + F.relu(1. + score_fake4).mean()
            real_loss = F.relu(1. - score_real2).mean() + F.relu(1. - score_real3).mean() + F.relu(1. - score_real4).mean()
            loss = fake_loss + real_loss
            return loss