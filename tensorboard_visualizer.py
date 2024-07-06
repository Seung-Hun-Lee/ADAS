from turtle import color
import torchvision.utils as vutils
import numpy as np
import torch
from PIL import Image
from utils.seg2edge import *
import matplotlib.pyplot as plt

# colour map
label_colours = [
    # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]]  # the color of ignored label(-1)
label_colours = list(map(tuple, label_colours))


def decode_labels(mask, num_images, num_classes=19):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[int(k_), int(j_)] = label_colours[int(k)]
        outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)

class Tensorboard_Visualizer:
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        self.num_classes = self.args.num_classes

    def visualize_imgs(self, imgs, iter, title='img', scale=1.0):
        x = vutils.make_grid(imgs.detach(), normalize=True, scale_each=True)
        self.writer.add_image('1_Images/%s'%title, scale*x, iter)
    
    def visualize_dict_imgs(self, imgs, iter, title='img', scale=1.0):
        imgs_ = torch.cat([imgs[key] for key in imgs.keys()], dim=0)
        x = vutils.make_grid(imgs_.detach(), normalize=True, scale_each=True)
        self.writer.add_image('1_Images/%s'%title, scale*x, iter)
    
    def visualize_labels(self, labels, iter, title='label'):
        x = decode_labels(labels.detach(), labels.size(0))
        x = vutils.make_grid(x, normalize=True, scale_each=True)
        self.writer.add_image('2_Labels/%s'%title, x, iter)

    def visualize_pred(self, seg_pred, iter, title='pred'):
        segmap = torch.argmax(seg_pred.detach(), dim=1)
        x = decode_labels(segmap, seg_pred.size(0))
        x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=self.args.bs_mult)
        self.writer.add_image('3_Prediction/%s'%title, x, iter)













