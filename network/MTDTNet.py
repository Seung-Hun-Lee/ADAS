import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from torch.nn.utils import spectral_norm
import functools
import numbers
from utils import *


def slice_patches(imgs, hight_slice=4, width_slice=4):
    b, c, h, w = imgs.size()
    h_patch, w_patch = int(h / hight_slice), int(w / width_slice)
    patches = imgs.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
    patches = patches.contiguous().view(b, c, -1, h_patch, w_patch)
    patches = patches.transpose(1,2)
    patches = patches.reshape(-1, c, h_patch, w_patch)
    return patches


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        GN = functools.partial(nn.GroupNorm, 4)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            GN(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            GN(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                GN(filters)
            )

    def forward(self, inputs):
        output = self.main(inputs)
        output += self.shortcut(inputs)
        return output


class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super(Encoder, self).__init__()
        mid_ch = out_ch // 2
        GN = functools.partial(nn.GroupNorm, 4)
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=4, stride=2, padding=1, bias=True),
            GN(mid_ch),
            nn.ReLU(True),
            ResidualBlock(mid_ch, mid_ch),
            ResidualBlock(mid_ch, mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        output = self.main(inputs)
        return output


class Style_Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super(Style_Encoder, self).__init__()
        mid_ch = out_ch // 2
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            ResidualBlock(mid_ch, mid_ch),
            ResidualBlock(mid_ch, mid_ch),
        )
        self.gamma = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )
        self.beta = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        output = self.main(inputs)
        gamma = self.gamma(output)
        beta = self.beta(output)
        return gamma, beta


class Generator(nn.Module):
    def __init__(self, in_ch=64, out_ch=3):
        super(Generator, self).__init__()
        mid_ch = in_ch // 2
        self.Decoder_Conv = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            ResidualBlock(mid_ch, mid_ch),
            ResidualBlock(mid_ch, mid_ch),
            spectral_norm(nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.Tanh()
        )
    def forward(self, x):
        out = self.Decoder_Conv(x)
        out = (1 + out) / 2  # [-1, 1] -> [0, 1]
        rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        out = transforms.Normalize(*rgb_mean_std)(out)
        return out


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.to_relu_4_2 = nn.Sequential()

        for x in range(25):
            self.to_relu_4_2.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.to_relu_4_2(x)


class Multi_Head_Discriminator(nn.Module):
    def __init__(self, args, in_ch=3, patch_ch=512, fc_ch=500):
        super(Multi_Head_Discriminator, self).__init__()
        num_target_domains = len(args.target_dataset)
        if isinstance(args.crop_size, numbers.Number):
            crop_size_h, crop_size_w = args.crop_size // 16, args.crop_size // 16
        else:
            crop_size_h, crop_size_w = args.crop_size[0] // 8, args.crop_size[1] // 16
        mid_ch = [patch_ch//8, patch_ch//4, patch_ch//2]
        self.Conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, mid_ch[0], kernel_size=4, stride=2, padding=1, bias=True)),  
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(mid_ch[0], mid_ch[1], kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.Patch = nn.Sequential(
            spectral_norm(nn.Conv2d(mid_ch[1], mid_ch[2], kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(mid_ch[2], patch_ch, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(patch_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)),
        )

        self.FC = nn.Sequential(
            spectral_norm(nn.Linear(crop_size_h*crop_size_w*mid_ch[1], fc_ch)),
            nn.ReLU(),
            spectral_norm(nn.Linear(fc_ch, num_target_domains))
        )

    def forward(self, inputs):
        conv_output = self.Conv(inputs)
        patch_output = self.Patch(conv_output)
        fc_output = self.FC(conv_output.view(conv_output.size(0), -1))
        return (patch_output, fc_output)


class Label_Embed(nn.Module):
    def __init__(self, size, out_ch=64):
        super().__init__()
        self.size = size
        self.embed = nn.Conv2d(1, out_ch, 1, 1, 0)
    
    def forward(self, seg):
        return self.embed(F.interpolate(seg.unsqueeze(1).float(), size=self.size, mode='nearest'))


class TAD(nn.Module):
    def __init__(self, ch, targets):
        super().__init__()
        self.IN = nn.InstanceNorm2d(ch)
        self.mlp_mean = nn.ModuleDict()
        self.mlp_std = nn.ModuleDict()
        for dset in targets:
            self.mlp_mean[dset] = nn.Linear(ch, ch)
            self.mlp_std[dset] = nn.Linear(ch, ch)

    def forward(self, feature, target_mean, target_std, target):
        return self.mlp_std[target](target_std).unsqueeze(-1).unsqueeze(-1)*self.IN(feature) + self.mlp_mean[target](target_mean).unsqueeze(-1).unsqueeze(-1)


class TAD_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, targets):
        super().__init__()
        mid_ch = min(in_ch, out_ch)
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, mid_ch, 3, 1, 1))
        self.D_adain1 = TAD(mid_ch, targets)
        self.conv2 = spectral_norm(nn.Conv2d(mid_ch, out_ch, 3, 1, 1))
        self.D_adain2 = TAD(out_ch, targets)
        
        self.conv_s = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
        self.D_adain_s = TAD(out_ch, targets)

    def forward(self, feature, target_mean, target_std, target):
        x_s = self.D_adain_s(self.conv_s(feature), target_mean, target_std, target)  # shortcut
        dx = self.conv1(feature)
        dx = self.conv2(F.relu(self.D_adain1(dx, target_mean, target_std, target)))
        dx = self.D_adain2(dx, target_mean, target_std, target)
        return F.relu(x_s + dx)


class Domain_Style_Transfer(nn.Module):
    def __init__(self, targets, ch=64, h=256, w=512):
        super().__init__()
        self.targets = targets
        self.n = nn.ParameterDict()  # the number of samples
        self.m = nn.ParameterDict()  # mean feature
        self.s = nn.ParameterDict()  # variance feature

        size = [1, ch, h, w]
        for dset in targets:
            self.n[dset] = nn.Parameter(torch.tensor(0), requires_grad=False)
            self.m[dset] = nn.Parameter(torch.zeros(size), requires_grad=False)
            self.s[dset] = nn.Parameter(torch.zeros(size), requires_grad=False)
        
        self.gamma_res1 = TAD_ResBlock(ch, ch, targets)
        self.gamma_res2 = TAD_ResBlock(ch, ch, targets)
        self.beta_res1 = TAD_ResBlock(ch, ch, targets)
        self.beta_res2 = TAD_ResBlock(ch, ch, targets)

    def forward(self, gamma, beta, target):
        # Domain mean, std
        target_mean = self.m[target].mean(dim=(0,2,3)).unsqueeze(0)
        target_std = ((self.s[target].mean(dim=(0,2,3)))/self.n[target]).sqrt()
        gamma_convert = self.gamma_res1(gamma, target_mean, target_std, target)
        gamma_convert = self.gamma_res2(gamma_convert, target_mean, target_std, target)
        beta_convert = self.gamma_res1(beta, target_mean, target_std, target)
        beta_convert = self.gamma_res2(beta_convert, target_mean, target_std, target)
        
        return gamma_convert, beta_convert
    
    def update(self, feature, target):
        self.n[target] += 1
        if self.n[target] == 1:
            self.m[target] = nn.Parameter(feature, requires_grad=False)
            self.s[target] = nn.Parameter((feature - self.m[target].mean(dim=(0,2,3), keepdim=True)) ** 2, requires_grad=False)
        else:
            prev_m = self.m[target].mean(dim=(0,2,3), keepdim=True)  # 1 x C x 1 x 1
            self.m[target] += (feature - self.m[target]) / self.n[target]  # B x C x H x W
            curr_m = self.m[target].mean(dim=(0,2,3), keepdim=True)  # 1 x C x 1 x 1
            self.s[target] += (feature - prev_m) * (feature - curr_m)  # B x C x H x W


class Multi_Target_Domain_Transfer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if isinstance(args.crop_size, numbers.Number):
            h, w = args.crop_size, args.crop_size
        else:
            h, w = args.crop_size[0], args.crop_size[1]
        # slice_h, slice_w = args.slice_h, args.slice_w
        self.dset = args.DT_source_dataset + args.DT_target_dataset
        self.src = self.dset[0]
        self.trg = self.dset[1:]
        self.E = Encoder(out_ch=args.mtdt_feat_ch)
        self.G = Generator(in_ch=args.mtdt_feat_ch)
        self.SE = Style_Encoder(out_ch=args.mtdt_feat_ch)
        self.Embed = Label_Embed((h//2, w//2), out_ch=args.mtdt_feat_ch)
        self.DST = Domain_Style_Transfer(self.trg, ch=args.mtdt_feat_ch, h=h//2, w=w//2)
        self.alpha_recon = 10
        self.alpha_adv = 1. / len(self.trg)

    def forward(self, imgs, src_gt, netD=None, netP=None, mode='None'):
        transferred_imgs = dict()
        if mode == 'dis':  # update D
            real, fake = dict(), dict()
            with torch.no_grad():
                gamma_src, beta_src = self.SE(imgs[self.src])
                for target in self.trg:
                    gamma_trg, beta_trg = self.DST(gamma_src, beta_src, target)
                    transferred_imgs[target] = self.G(gamma_trg*self.Embed(src_gt) + beta_trg)
            for target in self.trg:
                real[target] = netD(slice_patches(imgs[target]))
                fake[target] = netD(slice_patches(transferred_imgs[target]))
            return self.multi_dis_loss(real, fake)

        elif mode == 'gen':  # update all except D
            fake = dict()
            direct_recon, indirect_recon = dict(), dict()
            vgg_feature = dict()
            recon_loss, perceptual_loss = 0, 0
            # direct recon for building feature space
            for domain in self.dset:
                feature = self.E(imgs[domain])
                direct_recon[domain] = self.G(feature)
                recon_loss += self.recon_loss(imgs[domain], direct_recon[domain])
            # indirect recon for disentangling content and style
            gamma_src, beta_src = self.SE(imgs[self.src])
            indirect_recon[self.src] = self.G(gamma_src*self.Embed(src_gt)+beta_src)
            recon_loss += self.recon_loss(imgs[self.src], indirect_recon[self.src])
            # domain transfer
            vgg_feature[self.src] = netP(imgs[self.src])
            for target in self.trg:
                gamma_trg, beta_trg = self.DST(gamma_src, beta_src, target)
                transferred_imgs[target] = self.G(gamma_trg*self.Embed(src_gt) + beta_trg)
                fake[target] = netD(slice_patches(transferred_imgs[target]))
                vgg_feature[target] = netP(transferred_imgs[target])
            gen_loss = self.multi_gen_loss(fake)
            perceptual_loss = self.perceptual_loss(vgg_feature)
            return recon_loss, gen_loss, perceptual_loss

        elif mode == 'transfer':
            gamma_src, beta_src = self.SE(imgs)
            for target in self.trg:
                gamma_trg, beta_trg = self.DST(gamma_src, beta_src, target)
                transferred_imgs[target] = self.G(gamma_trg*self.Embed(src_gt) + beta_trg)
            return transferred_imgs

        else:  # for visualize (eval)
            direct_recon, indirect_recon = dict(), dict()
            for domain in self.dset:
                feature = self.E(imgs[domain])
                direct_recon[domain] = self.G(feature)
            gamma_src, beta_src = self.SE(imgs[self.src])
            indirect_recon[self.src] = self.G(gamma_src*self.Embed(src_gt)+beta_src)
            for target in self.trg:
                gamma_trg, beta_trg = self.DST(gamma_src, beta_src, target)
                transferred_imgs[target] = self.G(gamma_trg*self.Embed(src_gt) + beta_trg)
            return direct_recon, indirect_recon, transferred_imgs
    
    def update(self, imgs):
        with torch.no_grad():
            for target in self.trg:
                feature = self.E(imgs[target])
                self.DST.update(feature, target)

    def recon_loss(self, img, recon):
        return self.alpha_recon * F.l1_loss(img, recon)

    def multi_dis_loss(self, real, fake):
        patch_dis_loss, domain_clf_loss = 0, 0
        for domain_label, target in enumerate(self.trg):
            # Real
            patch_real, domain_real = real[target]
            patch_dis_loss += F.relu(1. - patch_real).mean()
            domain_clf_loss += F.cross_entropy(domain_real, domain_label*torch.ones(patch_real.size(0), device=patch_real.device).long())
            # Fake
            patch_fake, domain_fake = fake[target]
            patch_dis_loss += F.relu(1. + patch_fake).mean()
            domain_clf_loss += F.cross_entropy(domain_fake, domain_label*torch.ones(patch_fake.size(0), device=patch_fake.device).long())
        return self.alpha_adv * (patch_dis_loss + domain_clf_loss)

    def multi_gen_loss(self, fake):
        patch_gen_loss, domain_clf_loss = 0, 0
        for domain_label, target in enumerate(self.trg):
            # Fake
            patch_fake, domain_fake = fake[target]
            patch_gen_loss += (-patch_fake.mean())
            domain_clf_loss += F.cross_entropy(domain_fake, domain_label*torch.ones(patch_fake.size(0), device=patch_fake.device).long())
        return self.alpha_adv * (patch_gen_loss + domain_clf_loss)

    def perceptual_loss(self, feature):
        p_loss = 0
        for target in self.trg:
            p_loss += F.mse_loss(feature[self.src], feature[target])
        return p_loss
    
