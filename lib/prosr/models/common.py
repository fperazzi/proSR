from math import floor, ceil, log2
import numpy as np
import numpy.random as random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import collections


class Conv2d(nn.Module):
    """
    Convolution with alternative padding specified as 'padding_type'
    Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           padding_type='REFLECTION', dilation=1, groups=1, bias=True)
    if padding is not specified explicitly, compute padding = floor(kernel_size/2)
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__()
        p = 0
        conv_block = []
        kernel_size = args[2]
        dilation = kwargs.pop('dilation', 1)
        padding = kwargs.pop('padding', None)
        if padding is None:
            if isinstance(kernel_size, collections.Iterable):
                assert(len(kernel_size) == 2)
            else:
                kernel_size = [kernel_size] * 2

            padding = (floor((kernel_size[0]-1)/2), ceil((kernel_size[0]-1)/2),
                       floor((kernel_size[1]-1)/2), ceil((kernel_size[1]-1)/2))

        try:
            if kwargs['padding_type'] == 'REFLECTION':
                conv_block += [nn.ReflectionPad2d(padding), ]
            elif kwargs['padding_type'] == 'ZERO':
                p = padding
            elif kwargs['padding_type'] == 'REPLICATE':
                conv_block += [nn.ReplicationPad2d(padding), ]

        except KeyError as e:
            # use default padding 'REFLECT'
            conv_block += [nn.ReflectionPad2d(padding), ]
        except Exception as e:
            raise e

        conv_block += [nn.Conv2d(*args, padding=p, dilation=dilation, **kwargs)]
        self.conv = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """ResBlock"""

    def __init__(self, blocks, planes, res_factor=1, act_type='RELU', act_params=dict()):
        super(ResBlock, self).__init__()
        self.blocks = blocks
        self.res_factor = res_factor
        if self.blocks == 'brcbrc':
            self.m = nn.Sequential(nn.BatchNorm2d(planes),
                                   get_activation(act_type, act_params),
                                   Conv2d(planes, planes, 3),
                                   nn.BatchNorm2d(planes),
                                   get_activation(act_type, act_params),
                                   Conv2d(planes, planes, 3),
                                   )
        elif self.blocks == 'crc':
            self.m = nn.Sequential(Conv2d(planes, planes, 3),
                                   get_activation(act_type, act_params),
                                   Conv2d(planes, planes, 3),
                                   )
        elif self.blocks == 'cbrcb':
            self.m = nn.Sequential(Conv2d(planes, planes, 3),
                                   nn.BatchNorm2d(planes),
                                   get_activation(act_type, act_params),
                                   Conv2d(planes, planes, 3),
                                   nn.BatchNorm2d(planes),
                                   )

    def forward(self, x):
        return self.res_factor*self.m(x) + x


class PixelShuffleUpsampler(nn.Sequential):
    """Upsample block with pixel shuffle"""

    def __init__(self, ratio, planes, woReLU=True):
        super(PixelShuffleUpsampler, self).__init__()
        assert ratio == 3 or log2(ratio) == int(log2(ratio))
        layers = []
        for i in range(int(log2(ratio))):
            if ratio == 3:
                mul = 9
            else:
                mul = 4
            layers += [Conv2d(planes, mul * planes, 3),
                       nn.PixelShuffle(2)]
            if not woReLU:
                layers.append(nn.ReLU(inplace=True))

        self.m = nn.Sequential(*layers)


class CompressionBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(CompressionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = super(CompressionBlock, self).forward(x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


def get_activation(act_type, activation_params=dict()):
    act_type = act_type.upper()
    if act_type == 'RELU':
        return nn.ReLU(inplace=True)
    elif act_type == 'ELU':
        return nn.ELU(inplace=True, **activation_params)
    elif act_type == 'LRELU' or act_type == 'LEAKYRELU':
        return nn.LeakyReLU(inplace=True, **activation_params)
    elif act_type == 'SELU':
        return nn.SELU(inplace=True)
    else:
        raise NotImplementedError('{} is not implemented, available activations are: {}'.format(
            act_type, ', '.join(['ReLU', 'ELU', 'SELU', 'LeakyReLU (LReLU)'])))


class ToVggInput(nn.Module):
    """vgg input"""

    def __init__(self, orig_mean, orig_mul):
        super(ToVggInput, self).__init__()
        self.orig_mean = nn.Parameter(torch.Tensor(orig_mean).view(1, 3, 1, 1), requires_grad=False)
        self.orig_mul = nn.Parameter(torch.Tensor([orig_mul]), requires_grad=False)
        self.mean = nn.Parameter(torch.Tensor([[[0.485]], [[0.456]], [[0.406]]]),
            requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([[[0.229]], [[0.224]], [[0.225]]]),
            requires_grad=False)

    def forward(self, x):
        return (x / self.orig_mul + self.orig_mean - self.mean) / self.std


def dequantize(var, mul_img):
    noise = var.data.new(var.size())
    max_v = mul_img / 255.0 / 2
    noise.uniform_(0, max_v)
    noise.sub_(max_v/2)
    var = Variable(noise) + var
    return var


def init_weights(m):
    import numpy as np
    if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        # m.weight.data.normal_(0, sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        b, c, h, w = m.weight.data.size()
        f = ceil(w / 2)
        cen = (2 * f - 1 - f % 2) / (2.0 * f)
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        fil = (1 - np.abs(xv / f - cen)) * (1 - np.abs(yv / f - cen))
        fil = fil[np.newaxis, np.newaxis, ...]
        fil = np.repeat(fil, 3, 0)
        m.weight.data.copy_(torch.from_numpy(fil))


def flip(x, dim):
    xsize = x.size()
    x = x.contiguous()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1),
        ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
