# ADAS (CVPR 2022)
### Pytorch implementation of paper:
### [Seunghun Lee, Wonhyeok Choi, Changjae Kim, Minwoo Choi, Sunghoon Im, "ADAS: A Direct Adaptation Strategy for Multi-Target Domain Adaptive Semantic Segmentation", CVPR (2022)](https://arxiv.org/abs/2203.06811)
## Requirements
Pytorch >= 1.8.0
You need GPU with no less than 32G memory.
## Data Preparation
Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), [Cityscapes](https://www.cityscapes-dataset.com/), [IDD](https://idd.insaan.iiit.ac.in/), [Mapillary](https://www.mapillary.com/datasets)
## Folder Structure of Datasets
```
├── data
      ├── GTA5
            ├── GT
                   ├── 01_labels
                   ├── 02_labels
                   ├── ...
            ├── Images
                   ├── 01_images
                   ├── 02_images
                   ├── ...
      ├── Cityscapes
            ├── gtFine
                   ├── train
                   ├── val
            ├── leftImg8bit
                   ├── train
                   ├── val
      ├── IDD
            ├── gtFine
                   ├── train
                   ├── val
            ├── leftImg8bit
                   ├── train
                   ├── val
      ├── mapillary
            ├── training
                   ├── images
                   ├── labels
            ├── validation
                   ├── images
                   ├── labels
```
## Train
1. MTDT-Net (image to multi-target images)
```
python MTDT_train.py -D [datasets] -N [num_classes] --iter [itrations] --ex [experiment_name]
example) python MTDT_train.py -D G C I M -N 19 --iter 3000 --ex MTDT_19
```
You can see the translated images on tensorboard.
```
CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir tensorboard --bind_all
```
2. Pretraining (option)
```
python Source_Only.py -D G C I M -N 19 --iter 100000 --ex Source_Only_19
```
3. Domain Adaptatoin with MTDT-Net
```
python MTDT_DA.py -D G C I M -N 19 --iter 200000 --load_mtdt checkpoint/MTDT_19/3000/netMTDT.pth --ex MTDT_DA_19
(If you have pretrained model, you can add '--load_seg checkpoint/Source_Only_19/100000/netT.pth' option.)
```
4. Domain Adaptation with BARS
```
python BARS_DA.py -D G C I M -N 19 --iter 200000 --load_mtdt checkpoint/MTDT_19/3000/netMTDT.pth --load_seg checkpoint/MTDT_DA_19/200000/netT.pth --ex BARS_DA_19
```

## Test
```
python test.py -D G C I M -N 19 --load_seg [trained network]
```




more coming soon
