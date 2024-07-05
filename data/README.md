
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
