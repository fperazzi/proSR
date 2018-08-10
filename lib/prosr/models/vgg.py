from collections import OrderedDict
from torchvision import models

import torch
import torch.nn as nn


class ToVggInput(nn.Module):
    """vgg input"""

    def __init__(self, orig_mean, orig_mul):
        super(ToVggInput, self).__init__()
        self.orig_mean = nn.Parameter(
            torch.Tensor(orig_mean).view(1, 3, 1, 1), requires_grad=False)
        self.orig_mul = nn.Parameter(
            torch.Tensor([orig_mul]), requires_grad=False)
        self.mean = nn.Parameter(
            torch.Tensor([[[0.485]], [[0.456]], [[0.406]]]),
            requires_grad=False)
        self.std = nn.Parameter(
            torch.Tensor([[[0.229]], [[0.224]], [[0.225]]]),
            requires_grad=False)

    def forward(self, x):
        return (x / self.orig_mul + self.orig_mean - self.mean) / self.std


class Vgg16(nn.Module):
    def __init__(self,
                 orig_mean,
                 orig_mul,
                 upto=5,
                 mean_pool=False,
                 requires_grad=False):
        super(Vgg16, self).__init__()
        self.upto = upto
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        if mean_pool:
            # conv1-1 relu conv1-2 relu
            # maxp conv2-1 relu conv2-2 relu
            # maxp conv3-1 relu conv3-2 relu conv3-3 relu ...
            feature_ranges = [(0, 4), (5, 9), (10, 16), (17, 23), (24, 30)]
        else:
            feature_ranges = [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]

        layers = OrderedDict()
        layers['data_conversion'] = ToVggInput(orig_mean, orig_mul)
        for i in range(upto):
            features = []
            if mean_pool and i > 0:
                features.append(nn.AvgPool2d(2))
            for k in range(*feature_ranges[i]):
                features.append(vgg_pretrained_features[k])
            layers['relu_%d' % (i + 1)] = torch.nn.Sequential(*features)
        for k, v in layers.items():
            self.add_module(k, v)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, acquire=None):
        if acquire is None:
            acquire = [self.upto]
        x = self.data_conversion(x)
        output = []
        for i in range(1, self.upto + 1):
            x = getattr(self, 'relu_%d' % i)(x)
            if i in acquire:
                output.append(x)
        return output
