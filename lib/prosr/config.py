# /usr/bin/env python
# from enum import Enum
from easydict import EasyDict as edict
from enum import Enum

import copy


class phase(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


""" Configuration file."""

######### ProSR #########
prosr_params = \
    edict({
        'train': {
            'dataset': {
                'path': 'data/datasets/DIV2K/DIV2K_train_HR',
                'mean': [0.4488, 0.4371, 0.4040],  # mean value to extract from the (0, 1) image values
                'stddev': [0.0039215, 0.0039215, 0.0039215]  # multiply the image value by this factor, resulting value range of image [-127.5, 127.5]
            },
            'epochs': 600,  # different to paper
            'batch_size': 16,
            'growing_steps': [0.12, 0.25, 0.45, 0.6, 1.00],
            'lr_schedule_patience': 30,
            'lr': 0.0001,
            'lr_decay': 0.5,
            'smallest_lr': 1e-5,
            'l1_loss_weight': 1.0,
        },
        'G': {
            'max_scale': 8,
            'residual_denseblock': True,  # ProSR_l and ProSRGan uses residual links, ProSR_l doesn't
            # densenet hyperparameters
            'num_init_features': 160,
            'growth_rate': 40,
            # architecture for each pyramid level
            # len(level_config) means the number of pyramids (equals the number of scales)
            # len(level_config(i)) means the number of DCUs in pyramid i
            # level_config(i)(j) means the number of dense layers in j-th DCU of the i-th pyramid.
            'level_config': [[8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8], [8]],
            # maximum number of features before subpixel upsampling
            'max_num_feature': 312,
            'ps_woReLU': False,  # ps without relu
            'level_compression': -1,  # used between pyramid levels, if <0: compress to same as input
            'bn_size': 4,
            'res_factor': 0.2,  # scale residual
        },
        'test': {
            'dataset': {
                'path': 'data/datasets/DIV2K/DIV2K_valid_HR',
                'mean': [0.4488, 0.4371, 0.4040],  # mean value to extract from the (0, 1) image values
                'stddev': [0.0039215, 0.0039215, 0.0039215]  # multiply the image value by this factor, resulting value range of image [-127.5, 127.5]
            },
        },
        'data': {
            'input_size': [48, 36, 24],  # reduce input size for 4x and 8x to save memory
            'max_scale': 8,
            'scale': [2, 4, 8]
        }
    })
prosrs_params = copy.deepcopy(prosr_params)
prosrs_params.train.batch_size = 32
prosrs_params.G.level_config = [[6, 6, 6, 6], [6, 6], [6]]
prosrs_params.G.num_init_features = 24
prosrs_params.G.growth_rate = 12
prosrs_params.G.block_compression = 0.4
prosrs_params.G.level_compression = 0.5
prosrs_params.G.residual_denseblock = False
prosrs_params.G.res_factor = 1.0

prosrgan_params = copy.deepcopy(prosr_params)
prosrgan_params.D = edict({
    'which_epoch': 'latest',
    'which_model_netD': 'srgan',
    'input_residual': True,
    'scale_overhead': True,
    'warmup_epochs': 0,
    'update_freq': 2,
    'use_lsgan': True,
    'ndf': 64,
    'act_type': 'LRELU',
    'act_params': {
        'negative_slope': 0.2
    },
})
prosrgan_params.train.D_lr = 0.0001
prosrgan_params.train.vgg_loss_weight = [0.5, 2]
prosrgan_params.train.gan_loss_weight = 1
prosrgan_params.train.l1_loss_weight = 0
prosrgan_params.G.vgg = [2, 4]
prosrgan_params.G.vgg_mean_pool = True
