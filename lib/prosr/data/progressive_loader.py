import random
from PIL import Image
from collections import Iterable
import torch.utils.data as torch_data
import torchvision.transforms as transforms

from .. import Phase
from ..logger import error
from . import multiproc
from .util import *


class Dataset(object):
    """docstring for Dataset"""

    def __init__(self, phase, source, target, upscale_factor, crop_size, mean, stddev,
                 **kwargs):

        super(Dataset, self).__init__()
        self.phase = phase
        self.scale = upscale_factor if isinstance(upscale_factor, Iterable) else [upscale_factor]
        self.crop_size = [crop_size]*len(self.scale) if not isinstance(crop_size, Iterable) else crop_size
        assert len(self.crop_size) == len(self.scale)
        self.crop_size = dict([(s, size) for s, size in zip(self.scale, self.crop_size)])
        self.mean = mean
        self.stddev = stddev

        self.augment = self.phase == Phase.TRAIN

        self.image_loader = pil_loader

        if len(target) and len(source) and len(source) != len(target):
            error(
                "Inconsistent number of images! Found {} source image(s) and {} target image(s).".
                format(len(source), len(target)))
        elif self.phase == Phase.TRAIN:
            assert len(target), "Training requires target files"
        else:
            assert len(target) or len(source), "At least one of target and source is not specified."

        self.source_fns = source
        self.target_fns = target

        # Input normalization
        self.normalize_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])

    def __len__(self):
        return len(self.source_fns) or len(self.target_fns)

    def __getitem__(self, index):
        return self.get(index, random.choice(self.scale))

    def get(self, index, scale=None):
        if scale:
            assert scale in self.scale, "scale {}".format(scale)
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
        else:
            ret_data['input'] = downscale_by_ratio(ret_data['target'], scale, method=Image.BICUBIC)
            ret_data['input_fn'] = self.target_fns[index]

        # Crop image
        if self.crop_size[scale]:
            if self.phase == Phase.TRAIN:
                ret_data['target'], ret_data['input'] = random_crop_pairs(
                        self.crop_size[scale], scale, ret_data['target'], ret_data['input'])
            else:
                ret_data['target'], ret_data['input'] = center_crop(
                        self.crop_size[scale], scale, ret_data['target'], ret_data['input'])

        if self.augment:  # TODO FIX
            ret_data['input'], ret_data['target'] = augment_pairs(ret_data['input'], ret_data['target'])

        ret_data['bicubic'] = downscale_by_ratio(ret_data['input'], 1/scale, method=Image.BICUBIC)

        ret_data['input'] = self.normalize_fn(ret_data['input'])
        ret_data['bicubic'] = self.normalize_fn(ret_data['bicubic'])
        if len(self.target_fns):
            ret_data['target'] = self.normalize_fn(ret_data['target'])

        return ret_data


class DataLoader(multiproc._DataLoader):
    """Hacky way to progressively load scales"""

    def __init__(self, dataset, batch_size, upscale_factor=None):
        self.dataset = dataset
        # this keeps consistent with experiments in the paper
        if self.dataset.phase == Phase.TRAIN:
            self.dataset.target_fns = self.dataset.target_fns*batch_size
            self.dataset.source_fns = self.dataset.source_fns*batch_size

        self.phase = dataset.phase
        super(DataLoader, self).__init__(self.dataset,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=self.phase == Phase.TRAIN,
                num_workers=16,
                random_vars=dataset.scale if self.phase == Phase.TRAIN else None,
                drop_last=True,
                sampler=None)
