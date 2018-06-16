# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
import torch
import torch.nn as nn

from .common import get_activation, Conv2d


class _DenseLayer(nn.Sequential):
  def __init__(self, num_input_features, growth_rate, bn_size,
      activation, activation_params):
    super(_DenseLayer, self).__init__()
    num_output_features = bn_size * growth_rate

    self.add_module('conv_1', nn.Conv2d(num_input_features, num_output_features,
            kernel_size=1, stride=1,
            bias=True)),

    self.add_module('relu_2', get_activation(activation, activation_params)),
    self.add_module('conv_2', Conv2d(num_output_features, growth_rate,
            3, stride=1, bias=True)),

  def forward(self, x):
    new_features = super(_DenseLayer, self).forward(x)
    return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
  def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
      activation='ReLU', activation_params=dict()):
    super(_DenseBlock, self).__init__()
    for i in range(num_layers):
      layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
        bn_size, activation, activation_params)
      self.add_module('denselayer%d' % (i + 1), layer)
