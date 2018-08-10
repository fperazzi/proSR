from .layers import (_DenseBlock, CompressionBlock, Conv2d, DenseResidualBlock,
                     init_weights, PixelShuffleUpsampler, ResidualBlock)
from collections import OrderedDict
from enum import Enum
from math import log2
from prosr.logger import error

import torch.nn as nn


##############################################################################
# Classes
##############################################################################
class block_type(Enum):
    BRCBRC = 'BRCBRC'
    CRC = 'CRC'
    CBRCB = 'CBRCB'


class ProSR(nn.Module):
    """docstring for PyramidDenseNet"""

    def __init__(self, residual_denseblock, num_init_features, bn_size,
                 growth_rate, ps_woReLU, level_config, level_compression,
                 res_factor, max_num_feature, max_scale, **kwargs):
        super(ProSR, self).__init__()
        self.max_scale = max_scale
        self.n_pyramids = int(log2(self.max_scale))

        # used in curriculum learning, initially set to the last scale
        self.current_scale_idx = self.n_pyramids - 1

        self.residual_denseblock = residual_denseblock
        self.DenseBlock = _DenseBlock
        self.Upsampler = PixelShuffleUpsampler
        self.upsample_args = {'woReLU': ps_woReLU}

        denseblock_params = {
            'num_layers': None,
            'num_input_features': num_init_features,
            'bn_size': bn_size,
            'growth_rate': growth_rate,
        }

        num_features = denseblock_params['num_input_features']

        # Initiate network

        # each scale has its own init_conv
        for s in range(1, self.n_pyramids + 1):
            self.add_module('init_conv_%d' % s, Conv2d(3, num_init_features,
                                                       3))

        # Each denseblock forms a pyramid
        for i in range(self.n_pyramids):
            block_config = level_config[i]
            pyramid_residual = OrderedDict()

            # starting from the second pyramid, compress the input features
            if i != 0:
                out_planes = num_init_features if level_compression <= 0 else int(
                    level_compression * num_features)
                pyramid_residual['compression_%d' % i] = CompressionBlock(
                    in_planes=num_features, out_planes=out_planes)
                num_features = out_planes

            # serial connect blocks
            for b, num_layers in enumerate(block_config):
                denseblock_params['num_layers'] = num_layers
                denseblock_params['num_input_features'] = num_features
                # residual dense block used in ProSRL and ProSRGAN
                if self.residual_denseblock:
                    pyramid_residual['residual_denseblock_%d' %
                                     (b + 1)] = DenseResidualBlock(
                                         res_factor=res_factor,
                                         **denseblock_params)
                else:
                    block, num_features = self.create_denseblock(
                        denseblock_params,
                        with_compression=(b != len(block_config) - 1),
                        compression_rate=kwargs['block_compression'])
                    pyramid_residual['denseblock_%d' % (b + 1)] = block

            # conv before upsampling
            block, num_features = self.create_finalconv(
                num_features, max_num_feature)
            pyramid_residual['final_conv'] = block
            self.add_module('pyramid_residual_%d' % (i + 1),
                            nn.Sequential(pyramid_residual))

            # upsample the residual by 2 before reconstruction and next level
            self.add_module(
                'pyramid_residual_%d_residual_upsampler' % (i + 1),
                self.Upsampler(2, num_features, **self.upsample_args))

            # reconstruction convolutions
            reconst_branch = OrderedDict()
            out_channels = num_features
            reconst_branch['final_conv'] = Conv2d(out_channels, 3, 3)
            self.add_module('reconst_%d' % (i + 1),
                            nn.Sequential(reconst_branch))

        init_weights(self)

    def get_init_conv(self, idx):
        """choose which init_conv based on curr_scale_idx (1-based)"""
        return getattr(self, 'init_conv_%d' % idx)

    def forward(self, x, upscale_factor=None, blend=1.0):
        if upscale_factor is None:
            upscale_factor = self.max_scale
        else:
            valid_upscale_factors = [
                2**(i + 1) for i in range(self.n_pyramids)
            ]
            if upscale_factor not in valid_upscale_factors:
                error("Invalid upscaling factor {}: choose one of: {}".format(
                    upscale_factor, valid_upscale_factors))
                raise SystemExit(1)

        feats = self.get_init_conv(log2(upscale_factor))(x)
        for s in range(1, int(log2(upscale_factor)) + 1):
            if self.residual_denseblock:
                feats = getattr(self, 'pyramid_residual_%d' % s)(feats) + feats
            else:
                feats = getattr(self, 'pyramid_residual_%d' % s)(feats)
            feats = getattr(
                self, 'pyramid_residual_%d_residual_upsampler' % s)(feats)

            # reconst residual image if reached desired scale /
            # use intermediate as base_img / use blend and s is one step lower than desired scale
            if 2**s == upscale_factor or (blend != 1.0 and 2**
                                          (s + 1) == upscale_factor):
                tmp = getattr(self, 'reconst_%d' % s)(feats)
                # if using blend, upsample the second last feature via bilinear upsampling
                if (blend != 1.0 and s == self.current_scale_idx):
                    base_img = nn.functional.upsample(
                        tmp,
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True)
                if 2**s == upscale_factor:
                    if (blend != 1.0) and s == self.current_scale_idx + 1:
                        tmp = tmp * blend + (1 - blend) * base_img
                    output = tmp

        return output

    def create_denseblock(self,
                          denseblock_params,
                          with_compression=True,
                          compression_rate=0.5):
        block = OrderedDict()
        block['dense'] = self.DenseBlock(**denseblock_params)
        num_features = denseblock_params['num_input_features']
        num_features += denseblock_params['num_layers'] * denseblock_params['growth_rate']

        if with_compression:
            out_planes = num_features if compression_rate <= 0 else int(
                compression_rate * num_features)
            block['comp'] = CompressionBlock(
                in_planes=num_features, out_planes=out_planes)
            num_features = out_planes

        return nn.Sequential(block), num_features

    def create_finalconv(self, in_channels, max_channels=None):
        block = OrderedDict()
        if in_channels > max_channels:
            block['final_comp'] = CompressionBlock(in_channels, max_channels)
            block['final_conv'] = Conv2d(max_channels, max_channels, (3, 3))
            out_channels = max_channels
        else:
            block['final_conv'] = Conv2d(in_channels, in_channels, (3, 3))
            out_channels = in_channels
        return nn.Sequential(block), out_channels

    def class_name(self):
        return 'ProSR'


class EDSR(nn.Module):
    """
  Implementation of thstkdgus35/EDSR-PyTorch: PyTorch version of the paper
  'Enhanced Deep Residual Networks for Single Image Super-Resolution' (CVPRW 2017)
  """

    def __init__(self, upscale_factor, num_blocks=36, **kwargs):
        super(EDSR, self).__init__()

        if upscale_factor % 2 != 0 and upscale_factor != 1:
            error("Upscaling factor must be 1 or a multiple of 2")
            raise SystemExit(1)

        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor
        self.scales = [self.upscale_factor]

        # Projection from image space.
        self.add_module('init_conv', Conv2d(3, 256, 3))

        # Backbone
        arch = OrderedDict()
        for i in range(1, self.num_blocks + 1):
            arch['resblock_%d' % i] = ResidualBlock(
                block_type.CRC, 'RELU', 256, res_factor=0.1)
        arch['final_conv'] = Conv2d(256, 256, 3)
        self.add_module('residual', nn.Sequential(arch))

        # Upsampling and reconstruction
        self.add_module('upsampler', PixelShuffleUpsampler(
            upscale_factor, 256))
        self.add_module(
            'reconst',
            nn.Sequential(OrderedDict([('reconst_conv0', Conv2d(256, 3, 3))])))

    def forward(self, x, scale=None, blend=1):
        if scale is not None and scale != self.upscale_factor:
            error("Invalid upscaling factor: choose one of: {}".format(
                [self.upscale_factor]))

        init_conv = self.init_conv(x)
        residual = self.residual(init_conv)
        output = init_conv + residual
        output = self.upsampler(output)
        output = self.reconst(output)

        return output

    def class_name(self):
        return 'ESDR'
