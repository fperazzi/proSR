from . import multiproc
from .. import Phase
from ..logger import error
from .util import *
from collections import Iterable
from PIL import Image

import copy
import random
import torchvision.transforms as transforms


class Dataset(object):
    """docstring for Dataset"""

    def __init__(self, phase, source, target, scale, input_size, mean,
                 stddev, downscale, **kwargs):

        super(Dataset, self).__init__()
        self.phase = phase
        self.scale = scale if isinstance(
            scale, Iterable) else [scale]
        self.input_size = [input_size] * len(self.scale) if not isinstance(
            input_size, Iterable) else input_size
        if phase == Phase.TRAIN:
            assert len(self.input_size) == len(self.scale)
        self.input_size = dict(
            [(s, size) for s, size in zip(self.scale, self.input_size)])
        self.mean = mean
        self.stddev = stddev

        self.augment = self.phase == Phase.TRAIN

        self.image_loader = pil_loader
        self.downscale=downscale

        if len(target) and len(source) and len(source) != len(target):
            error(
                "Inconsistent number of images! Found {} source image(s) and {} target image(s).".
                format(len(source), len(target)))
        elif self.phase == Phase.TRAIN:
            assert len(target), "Training requires target files"
        else:
            assert len(target) or len(
                source), "At least one of target and source is not specified."

        self.source_fns = source
        self.target_fns = target

        # In testing and validation phase, append fns with target scales
        if self.phase != Phase.TRAIN:
            if len(self.source_fns) > 0:
                self.source_fns = self.source_fns * len(self.scale)
            if len(self.target_fns) > 0:
                self.target_fns = self.target_fns * len(self.scale)

        # Input normalization
        self.normalize_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])

    def __len__(self):
        return len(self.source_fns) or len(self.target_fns)

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, scale=None):
        if scale:
            assert scale in self.scale, "scale {}".format(scale)
        elif self.phase != Phase.TRAIN:
            scale = self.scale[index % len(self.scale)]
        else:
            scale = random.choice(self.scale)

        ret_data = {}
        ret_data['scale'] = scale

        # Load target image
        if len(self.target_fns):
            target_img = self.image_loader(self.target_fns[index])
            w, h = target_img.size
            target_img = target_img.crop((0, 0, w - w % scale, h - h % scale))
            ret_data['target'] = target_img
            ret_data['target_fn'] = self.target_fns[index]

        # Load input image
        if len(self.source_fns):
            ret_data['input'] = self.image_loader(self.source_fns[index])
            ret_data['input_fn'] = self.source_fns[index]

            if self.downscale:
                ret_data['input'] = downscale_by_ratio(
                    ret_data['input'], scale, method=Image.BICUBIC)
        else:
            ret_data['input'] = downscale_by_ratio(
                ret_data['target'], scale, method=Image.BICUBIC)
            ret_data['input_fn'] = self.target_fns[index]

        # Crop image
        if self.input_size[scale]:
            if self.phase == Phase.TRAIN:
                ret_data['target'], ret_data['input'] = random_crop_pairs(
                    self.input_size[scale], scale, ret_data['target'],
                    ret_data['input'])
            else:
                ret_data['target'], ret_data['input'] = center_crop(
                    self.input_size[scale], scale, ret_data['target'],
                    ret_data['input'])

        if self.augment:  # TODO FIX
            ret_data['input'], ret_data['target'] = augment_pairs(
                ret_data['input'], ret_data['target'])

        ret_data['bicubic'] = downscale_by_ratio(
            ret_data['input'], 1 / scale, method=Image.BICUBIC)

        ret_data['input'] = self.normalize_fn(ret_data['input'])
        ret_data['bicubic'] = self.normalize_fn(ret_data['bicubic'])
        if len(self.target_fns):
            ret_data['target'] = self.normalize_fn(ret_data['target'])

        return ret_data


class DataLoader(multiproc.MyDataLoader):
    """Hacky way to progressively load scales"""

    def __init__(self, dataset, batch_size, scale=None):
        self.dataset = dataset
        # this keeps consistent with experiments in the paper
        if self.dataset.phase == Phase.TRAIN:
            self.dataset.target_fns = self.dataset.target_fns * batch_size
            self.dataset.source_fns = self.dataset.source_fns * batch_size
        self.phase = dataset.phase

        super(DataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=(self.phase == Phase.TRAIN),
            num_workers=16,
            random_vars=copy.deepcopy(dataset.scale) if self.phase == Phase.TRAIN else None,
            drop_last=True,
            sampler=None)

