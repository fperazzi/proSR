import random
from math import floor

from PIL import Image

import torchvision.transforms as transforms
from prosr import config
from prosr.logger import error

from . import multiproc
from .util import *


class Dataset(object):
    """docstring for Dataset"""

    def __init__(self, phase, source, target, upscale_factor, crop_size, mean, stddev,
                 **kwargs):

        super(Dataset, self).__init__()
        self.phase = phase

        self.augment = self.phase == config.phase.TRAIN

        self.image_loader = pil_loader

        self.crop_size = crop_size
        self.mean = mean
        self.stddev = stddev

        if len(target) and len(source) and len(source) != len(target):
            error(
                "Inconsistent number of images! Found {} source image(s) and {} target image(s).".
                format(len(source), len(target)))
        else:
            assert len(target) or len(source), "At least one of target and source is not specified."

        self.source_fns = source
        self.target_fns = target

        # TODO: hardcode scale
        if self.phase == config.phase.TRAIN:
            self.scale = [2, 4, 8]
        else:
            self.scale = [upscale_factor]

        # Input normalization
        self.normalize_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, index):
        return self.get(index, random.choice(self.scale))

    def get(self, index, scale=None):
        assert scale in self.scale, "scale {}".format(scale)
        ret_data = {}
        ret_data['scale'] = scale
        input_img = self.image_loader(self.source_fns[index])

        # Load target image
        if len(self.target_fns):
            target_img = self.image_loader(self.target_fns[index])
            w, h = target_img.size
            target_img = target_img.crop((0, 0, w - w % scale, h - h % scale))
            ret_data['target'] = target_img
            ret_data['target_fn'] = self.target_fns[index]

        # Load input image
        if len(self.source_fns):
            ret_data['input'] = input_img
            ret_data['input_fn'] = self.source_fns[index]
        else:
            ret_data['input'] = downscale_by_ratio(ret_data['target'], scale, method=Image.BICUBIC)
            ret_data['input_fn'] = self.target_fns[index]

        # Crop image
        if self.crop_size:
            ret_data['target'], ret_data['input'] = random_crop_pairs(
                    self.crop_size, scale, ret_data['target'], ret_data['input'])

        if self.augment:  # TODO FIX
            ret_data['input'], ret_data['target'] = augment_pairs(ret_data['input'], ret_data['target'])

        ret_data['bicubic'] = downscale_by_ratio(ret_data['input'], 1/scale, method=Image.BICUBIC)

        ret_data['input'] = self.normalize_fn(ret_data['input'])
        ret_data['bicubic'] = self.normalize_fn(ret_data['bicubic'])
        if len(self.target_fns):
            ret_data['target'] = self.normalize_fn(ret_data['target'])

        return ret_data


class DataLoader(object):
    """docstring for DataLoader"""

    def __init__(self, dataset, batch_size, upscale_factor=None):
        super(DataLoader, self).__init__()

        self.phase = dataset.phase
        self.dataset = dataset
        self.batch_size = batch_size
        self.scale = self.dataset.scale

        if self.phase == config.phase.TEST and \
                batch_size != 1:
            error('Batch size must be one during test')

        self._dataloader = self.create_loader()

    def create_loader(self):
        return multiproc._DataLoader(
            self.dataset,
            pin_memory=True,
            batch_size=self.batch_size,  #Fix
            shuffle=self.phase == config.phase.TRAIN,
            num_workers=16,
            # make sure to copy by value
            random_vars=self.scale,
            drop_last=True,
            sampler=None)

    def __iter__(self):
        return self._dataloader.__iter__()
