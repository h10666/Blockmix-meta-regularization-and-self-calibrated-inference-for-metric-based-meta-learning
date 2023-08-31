# coding=utf-8
import os, glob
import shutil
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import skimage.transform
import scipy.stats as stats
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

patch_xl = np.array([0,0,0,74,74,74,148,148,148])
patch_xr = np.array([74,74,74,148,148,148,224,224,224])
patch_yl = np.array([0,74,148,0,74,148,0,74,148])
patch_yr = np.array([74,148,224,74,148,224,74,148,224])


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def jigsaw_aug(image, auxiliary_images, min_replace_block_num=0, max_replace_block_num=2):
    aug_images = []
    for i in range(auxiliary_images.size(0)):
        replace_block_num = np.random.randint(min_replace_block_num, max_replace_block_num + 1)
        replaced_indexs = np.random.choice(9, replace_block_num, replace=False)

        aug_image = image.clone()
        aux_img = auxiliary_images[i,:,:,:]

        for l in range(replace_block_num):
            replaced_index = int(replaced_indexs[l])
            aug_image[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]] = aux_img[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]]
        aug_images.append(aug_image)
    aug_images=torch.stack(aug_images,dim=0)
    return aug_images


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count