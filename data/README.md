
## Data Preparation
Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), [Cityscapes](https://www.cityscapes-dataset.com/), [IDD](https://idd.insaan.iiit.ac.in/), [Mapillary](https://www.mapillary.com/datasets)

We adopt Class uniform sampling proposed in [this paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf) to handle class imbalance problems. [GTAVUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/gtav.py#L306) and [CityscapesUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/cityscapes.py#L324) are the datasets to which Class Uniform Sampling is applied.

#### We used [GTAV_Split](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip) to split GTAV dataset into training/validation/test set. Please refer the txt files in [split_data](https://github.com/shachoi/RobustNet/tree/main/split_data).

#### You should modify the path in **"<path_to_robustnet>/config.py"** according to your dataset path.
```
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = <YOUR_CITYSCAPES_PATH>
#IDD Dataset Dir Location
__C.DATASET.IDD_DIR = <YOUR_IDD_PATH>
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = <YOUR_MAPILLARY_PATH>
#GTAV Dataset Dir Location
__C.DATASET.GTAV_DIR = <YOUR_GTAV_PATH>
```

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
