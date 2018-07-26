import collections
import glob
import os.path as osp
from math import floor

from PIL import Image

import torchvision.transforms as transforms
from prosr import config
from prosr.logger import error

from . import multiproc
from .util import *


class Dataset(object):
    """docstring for Dataset"""

    def __init__(self, phase, source, target, crop_size, mean, stddev,
                 **kwargs):

        super(Dataset, self).__init__()
        self.phase = phase

        self.augment = self.phase == config.phase.TRAIN

        self.image_loader = pil_loader

        self.crop_size = crop_size
        self.mean = mean
        self.stddev = stddev

        if len(target) and len(source) != len(target):
            error(
                "Inconsistent number of images! Found {} source image(s) and {} target image(s).".
                format(len(source), len(target)))

        self.source_fns = source
        self.target_fns = target

        # Dataset augumentation
        self.augment_fn = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Lambda(lambda img: random_rot90(img))
        ])

        # Input normalization
        self.normalize_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index):

        ret_data = {}
        input_img = self.image_loader(self.source_fns[index])

        # Load input image
        ret_data['input'] = input_img
        ret_data['input_fn'] = self.source_fns[index]

        # Load target image
        if len(self.target_fns):
            target_img = self.image_loader(self.target_fns[index])
            ret_data['target'] = target_img
            ret_data['target_fn'] = self.target_fns[index]

        # Crop image
        if self.crop_size:
            scale = int(
                round(ret_data['target'].size[0] / ret_data['input'].size[0]))

            ret_data['target'], ret_data['input'] = random_crop_pairs(
                self.crop_size, scale, ret_data['target'], ret_data['input'])

        ret_data['input'] = self.normalize_fn(ret_data['input'])
        if len(self.target_fns):
            ret_data['target'] = self.normalize_fn(ret_data['target'])

        if self.augment:  #TODO FIX
            ret_data['input'] = self.augment_fn(ret_data['input'])
            if len(self.target_fns):
                ret_data['target'] = self.augment_fn(ret_data['target'])

        return ret_data


class DataLoader(object):
    """docstring for DataLoader"""

    def __init__(self, dataset, batch_size):
        super(DataLoader, self).__init__()

        self.phase = dataset.phase
        self.dataset = dataset
        self.batch_size = batch_size


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
            num_workers=1,
            # make sure to copy by value
            random_vars=[],
            drop_last=True,
            sampler=None)

    def __iter__(self):
        return self._dataloader.__iter__()
