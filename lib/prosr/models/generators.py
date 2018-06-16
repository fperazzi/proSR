import torch.nn as nn
from collections import OrderedDict
from math import log2
from . import densenet
from .common import Conv2d, PixelShuffleUpsampler, \
  CompressionBlock, init_weights

##############################################################################
# Classes
##############################################################################

# Defines the generator using dense connections
class DenseNetGenerator(nn.Module):
  """
  EDSR stype: predict feature residual instead of image residual,
  adapt initial feature to final feature size, NN upsample initial feature
  """

  def __init__(self, opt):
    super(DenseNetGenerator, self).__init__()
    denseblock_params = {
      'num_layers': None,
      'activation': opt.act_type,
      'activation_params': opt.act_params.__dict__,
      'num_input_features': opt.num_init_features,
      'bn_size': opt.bn_size,
      'growth_rate': opt.growth_rate,
    }

    self.DenseBlock = densenet._DenseBlock

    self.Upsampler = PixelShuffleUpsampler
    self.upsample_args = {'woReLU': opt.ps_woReLU}

    self.initiate(opt, denseblock_params)

  def initiate(self, opt, denseblock_params):
    pass

  def create_denseblock(self, denseblock_params, with_compression=True, compression_rate=0.5):
    block = OrderedDict()
    block['dense'] = self.DenseBlock(**denseblock_params)
    num_features = denseblock_params['num_input_features']
    num_features += denseblock_params['num_layers'] * denseblock_params['growth_rate']

    if with_compression:
      out_planes = num_features if compression_rate <= 0 \
        else int(compression_rate * num_features)
      block['comp'] = CompressionBlock(in_planes=num_features,
                  out_planes=out_planes)
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

  def forward(self, x, scale=None, blend=1):
    pass


class ProSR(DenseNetGenerator):
  """docstring for PyramidDenseNet"""

  def __init__(self, opt,scale):
    self.opt = opt
    self.opt.scale = scale
    super().__init__(opt)

  def initiate(self, opt, denseblock_params):
    num_features = opt.num_init_features

    # each scale has its own init_conv
    for s in opt.scale:
      self.add_module('init_conv_%d' % int(log2(s)),
        Conv2d(opt.num_img_channels, opt.num_init_features, 3))

    # Each denseblock forms a pyramid
    for i in range(int(log2(max(opt.scale)))):
      block_config = opt.level_config[i]
      pyramid_residual = OrderedDict()

      # next scale feature
      if i != 0:
        out_planes = opt.num_init_features if opt.level_compression <= 0 \
          else int(opt.level_compression * num_features)
        comp = CompressionBlock(in_planes=num_features,
                    out_planes=out_planes)
        pyramid_residual['compression_%d' % i] = comp
        num_features = out_planes

      for b, num_layers in enumerate(block_config):
        denseblock_params['num_layers'] = num_layers
        denseblock_params['num_input_features'] = num_features

        if opt.residual_denseblock:
          block = DenseResidualBlock(**denseblock_params, res_factor=opt.res_factor)
          pyramid_residual['residual_denseblock_%d' % (b + 1)] = block
        else:
          block, num_features = self.create_denseblock(denseblock_params,
            with_compression=(b != len(block_config) - 1),
            compression_rate=opt.block_compression)
          pyramid_residual['denseblock_%d' % (b + 1)] = block

      # conv before upsampling
      block, num_features = self.create_finalconv(num_features, opt.max_num_feature)
      pyramid_residual['final_conv'] = block
      self.add_module('pyramid_residual_%d' % (i + 1), nn.Sequential(pyramid_residual))

      # upsample the residual by 2 before reconstruction and next level
      self.add_module('pyramid_residual_%d_residual_upsampler' % (i + 1),
        self.Upsampler(2, num_features, **self.upsample_args))

      # reconstruction convolutions
      reconst_branch = OrderedDict()
      out_channels = num_features
      reconst_branch['final_conv'] = Conv2d(out_channels, opt.num_img_channels, 3)
      self.add_module('reconst_%d' % (i + 1), nn.Sequential(reconst_branch))

  def get_init_conv(self, scale):
    idx = log2(scale)
    return getattr(self, 'init_conv_%d' % idx)

  def forward(self, x, scale=None, base_img=None, blend=1.0):
    try:
      max_scale_idx = self.max_scale_idx
    except AttributeError:
      max_scale_idx = len(self.opt.scale) - 1

    if scale is None:
      scale = self.opt.scale[max_scale_idx]

    feats = self.get_init_conv(scale)(x)
    output = []
    for s in range(1, int(log2(scale))+1):
      if self.opt.residual_denseblock:
        feats = getattr(self, 'pyramid_residual_%d' % s)(feats)+feats
      else:
        feats = getattr(self, 'pyramid_residual_%d' % s)(feats)
      feats = getattr(self, 'pyramid_residual_%d_residual_upsampler' % s)(feats)

      # reconst residual image if intermediate output is required / reached desired scale /
      # use intermediate as base_img / use blend and s is one step lower than desired scale
      if 2 ** s == scale or (blend != 1.0 and 2 ** (s+1) == scale):
        tmp = getattr(self, 'reconst_%d' % s)(feats)
        # if using blend, upsample the second last feature via bilinear upsampling
        if (blend != 1.0 and s == max_scale_idx):
          base_img = nn.functional.upsample(tmp, scale_factor=2, mode='bilinear')
        if 2 ** s == scale:
          if (blend != 1.0) and s == max_scale_idx + 1:
            tmp = tmp * blend + (1 - blend) * base_img
          output += [tmp]

    if not self.training:
      assert len(output) == 1
      output = output.pop()
    else:
      assert len(output) == 1
    return output


class DenseResidualBlock(nn.Sequential):
  def __init__(self, **kwargs):
    super(DenseResidualBlock, self).__init__()
    self.res_factor = kwargs.pop('res_factor')

    self.dense_block = densenet._DenseBlock(**kwargs)
    num_features = kwargs['num_input_features'] + kwargs['num_layers'] * kwargs['growth_rate']

    self.comp = CompressionBlock(in_planes=num_features,
                   out_planes=kwargs['num_input_features'],
                   )

  def forward(self, x, identity_x=None):
    if identity_x is None:
      identity_x = x
    return self.res_factor * super(DenseResidualBlock, self).forward(x) + identity_x
