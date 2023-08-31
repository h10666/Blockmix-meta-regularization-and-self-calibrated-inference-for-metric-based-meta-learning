# Blockmix: meta regularization and self-calibrated inference for metric-based meta-learning 


This repository provides code for "_**Blockmix: meta regularization and self-calibrated inference for metric-based meta-learning**_" ACM Multimedia 2020. 


## Requirements

 - `Python 3.6`
 - [`Pytorch`](http://pytorch.org/) >= 0.4.1 
 - `Torchvision` = 0.10
 - `scikit-image` = 0.18.1


## Data Preparation


>  [mini-ImageNet](https://github.com/twitter/meta-learning-lstm) 

>  [CUB_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) 

## How to run

```bash

python train.py --dataset [type of dataset] --model [backbone] --num_classes [num-classes] --nExemplars [num-shots]
python test.py --dataset CUB-200-2011 --model ResNet12 --num_classes 100 --nExemplars 5

# Example: run on CUB dataset, ResNet-12 backbone, 5-way 1-shot
python train.py --dataset CUB-200-2011 --model ResNet12 --num_classes 100 --nExemplars 1
python test.py --dataset CUB-200-2011 --model ResNet12 --num_classes 100 --nExemplars 1

```

## Citation
Please cite our paper if you find the work useful, thanks!
  ```bibtex
     @inproceedings{TangLPT20,
                    author    = {Hao Tang and Zechao Li and Zhimao Peng and Jinhui Tang},
                    title     = {BlockMix: Meta Regularization and Self-Calibrated Inference for Metric-Based Meta-Learning},
                    booktitle = {{ACM} Multimedia},
                    pages     = {610--618},
                    publisher = {{ACM}},
                    year      = {2020}
                    }
  ```



