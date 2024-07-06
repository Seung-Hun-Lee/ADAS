# ADAS (CVPR 2022)
### Pytorch implementation of paper:
### [Seunghun Lee, Wonhyeok Choi, Changjae Kim, Minwoo Choi, Sunghoon Im, "ADAS: A Direct Adaptation Strategy for Multi-Target Domain Adaptive Semantic Segmentation", CVPR (2022)](https://arxiv.org/abs/2203.06811)


<p align="center">
<img src="https://github.com/Seung-Hun-Lee/ADAS/assets/75882468/e9e8bdac-9a87-4ec8-b162-cfa7bb1df296" width="600">
</p>


## Requirements
Pytorch >= 1.8.0
You need GPU with no less than 32G memory.

## Installation

```bash
conda create --name adas python=3.8 -y
conda activate adas
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardX scikit-image==0.19 tqdm matplotlib
```


## Getting Started
See [Preparing Datasets for ADAS](data/README.md).

See [Getting Started with ADAS](GETTING_STARTED.md).


### Improvement (1)
We regard the filtered pixels by BARS as hard samples and progressively learn a greater number of these hard samples each epoch. The option to activate this feature is '--curriculum', and it is controlled by '--incremental_ratio'.

### Improvement (2)
We extended the [DACS](https://arxiv.org/abs/2007.08702) method for self-training using BARS. Refer to [Domain_mixer.py](network/domain_mixer.py) for details.


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





