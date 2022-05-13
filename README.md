# ADAS (CVPR 2022)
### Pytorch implementation of paper:
### [Seunghun Lee, Wonhyeok Choi, Changjae Kim, Minwoo Choi, Sunghoon Im, "ADAS: A Direct Adaptation Strategy for Multi-Target Domain Adaptive Semantic Segmentation", CVPR (2022)](https://arxiv.org/abs/2203.06811)
## Requirements
```
Pytorch 1.8.0
CUDA 11.1
python 3.8.10
numpy 1.21.0
scipy 1.7.1
tensorboardX
prettytable
```
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
            ├── GT
                   ├── train
                   ├── val
                   ├── test
            ├── Images
                   ├── train
                   ├── val
                   ├── test

      
├── data_list
      ├── GTA5
              ├── train_imgs.txt
              ├── train_labels.txt
      ├── Cityscapes
              ├── train_imgs.txt
              ├── val_imgs.txt
              ├── train_labels.txt
              ├── val_labels.txt


```
## Train
You must input the datasets(G, C, I, M), and experiment name.
```
python train.py -D [datasets] --ex [experiment_name]
example) python train.py -D G C I --ex G2CI
```
## Test
Input the same experiment_name that you trained and specific iteration.
```
python test.py -D [datasets] --ex [experiment_name (that you trained)] --load_step [specific iteration]
example) python test.py -D G C I --ex G2CI --load_step 100000
```
## Tensorboard
You can see all the results of each experiment on tensorboard.
```
CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir tensorboard --bind_all
```


more coming soon
