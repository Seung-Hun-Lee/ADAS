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





