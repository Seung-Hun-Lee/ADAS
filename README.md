# ADAS (CVPR 2022)
### Pytorch implementation of paper:
### [Seunghun Lee, Wonhyeok Choi, Changjae Kim, Minwoo Choi, Sunghoon Im, "ADAS: A Direct Adaptation Strategy for Multi-Target Domain Adaptive Semantic Segmentation", CVPR (2022)](https://arxiv.org/abs/2203.06811)
## Requirements
Pytorch >= 1.8.0
You need GPU with no less than 24G memory.

## Installation

```bash
conda create --name adas python=3.8 -y
conda activate adas
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardX scikit-image tqdm matplotlib


git clone git@https://github.com/Seung-Hun-Lee/ADAS.git
cd ADAS
pip install -r requirements.txt
```

## Data Preparation
Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), [Cityscapes](https://www.cityscapes-dataset.com/), [IDD](https://idd.insaan.iiit.ac.in/), [Mapillary](https://www.mapillary.com/datasets)
## Folder Structure of Datasets
You can set dataset roots in config.py. 
```
├── data
      ├── GTA5
            ├── images
                  ├── train
                         ├── 01
                         ├── 02
                         ├── ...
            ├── labels
                  ├── train
                         ├── 01
                         ├── 02
                         ├── ...
      ├── Cityscapes
            ├── leftImg8bit_trainvaltest
                  ├── leftImg8bit
                         ├── train
                         ├── val
            ├── gtFine_trainvaltest
                  ├── gtFine
                         ├── train
                         ├── val
      ├── IDD
            ├── leftImg8bit_trainvaltest
                  ├── leftImg8bit
                         ├── train
                         ├── val
            ├── gtFine_trainvaltest
                  ├── gtFine
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


1. Source only
```
sh scripts
```


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


## Citation
```BibTeX
@inproceedings{lee2022adas,
  title={Adas: A direct adaptation strategy for multi-target domain adaptive semantic segmentation},
  author={Lee, Seunghun and Choi, Wonhyeok and Kim, Changjae and Choi, Minwoo and Im, Sunghoon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19196--19206},
  year={2022}
}
```





