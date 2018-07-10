# /usr/bin/env python
import os

""" Configuration file."""

import os
import os.path as osp

import sys
from easydict import EasyDict as edict

from enum import Enum

class phase(Enum):
    TRAIN    = 'train'
    VAL      = 'val'
    TEST     = 'test'

__C = edict()

# Public access to configuration settings
defaults = __C

__C.train = edict()
__C.test  = edict()
__C.eval  = edict()

__C.G = edict()
__C.D = edict()
__C.R = edict()

__C.phase = phase.TEST

# Network scales
__C.max_scale= 8

__C.mean_img= (0.4488,0.4371,0.4040)  # mean value to extract from the (0, 1) image values
__C.mul_img= 255  # multiply the image value by this factor, resulting value range of image [-127.5, 127.5]

# directory to save and load model file during training and test
__C.image_mode= 'RGB'

__C.G.output_residual= True

# selects model to use for netG
__C.G.residual_denseblock = True  # ProSR_l and ProSRGan uses residual links
__C.G.num_img_channels= 3

# densenet hyperparameters
__C.G.num_init_features= 160
__C.G.growth_rate= 40

# architecture for each pyramid level
# len(level_config) means the number of pyramids (equals the number of scales)
# len(level_config(i)) means the number of DCUs in pyramid i
# level_config(i)(j) means the number of dense layers in j-th DCU of the i-th pyramid.
__C.G.level_config= [[8,8,8,8,8,8,8,8,8], [8,8,8], [8]]
__C.G.max_num_feature= 312

# if > 1 use denseblocks recursively
__C.G.act_type= 'ReLU'  # activation to be used
__C.G.act_params= dict()
__C.G.ps_woReLU= False
__C.G.level_compression= -1  # used between pyramid levels, if <0: compress to same as input
__C.G.bn_size= 4
__C.G.res_factor= 0.2  # scale residual

# upsample method for global skip connection with pillow
__C.G.upsample_method= 'BICUBIC'
