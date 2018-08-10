from math import floor
from numpy import random
from PIL import Image

import numpy as np


def random_rot90(img, r=None):
    if r is None:
        r = random.random() * 4  # TODO Check and rewrite func
    if r < 1:
        return img.transpose(Image.ROTATE_90)
    elif r < 2:
        return img.transpose(Image.ROTATE_270)
    elif r < 3:
        return img.transpose(Image.ROTATE_180)
    else:
        return img


def augment_pairs(img1, img2):
    vflip = random.random() > 0.5
    hflip = random.random() > 0.5
    rot = random.random() * 4
    if hflip:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
        img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)

    img1 = random_rot90(img1, rot)
    img2 = random_rot90(img2, rot)
    return img1, img2


def center_crop(crop_size, scale, hr, lr):
    oh_lr = ow_lr = crop_size
    oh_hr = ow_hr = oh_lr * scale
    w_lr, h_lr = lr.size
    offx_lr = (w_lr - crop_size) // 2
    offy_lr = (h_lr - crop_size) // 2
    offy_hr, offx_hr = int(offy_lr * scale), int(offx_lr * scale)
    return (hr.crop((offx_hr, offy_hr, offx_hr + ow_hr, offy_hr + oh_hr)),
            lr.crop((offx_lr, offy_lr, offx_lr + ow_lr, offy_lr + oh_lr)))


def random_crop_pairs(crop_size, scale, hr, lr):
    oh_lr = ow_lr = crop_size
    oh_hr = ow_hr = oh_lr * scale
    imw_lr, imh_lr = lr.size

    x0 = 0
    x1 = imw_lr - ow_lr + 1

    y0 = 0
    y1 = imh_lr - oh_lr + 1

    offy_lr = random.randint(y0, y1)
    offx_lr = random.randint(x0, x1)
    offy_hr, offx_hr = int(offy_lr * scale), int(offx_lr * scale)

    return (hr.crop((offx_hr, offy_hr, offx_hr + ow_hr, offy_hr + oh_hr)),
            lr.crop((offx_lr, offy_lr, offx_lr + ow_lr, offy_lr + oh_lr)))


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def downscale_by_ratio(img, ratio, method=Image.BICUBIC):
    if ratio == 1:
        return img
    w, h = img.size
    w, h = floor(w / ratio), floor(h / ratio)
    return img.resize((w, h), method)
