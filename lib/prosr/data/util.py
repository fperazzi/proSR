import random

import numpy as np
from PIL import Image


def random_rot90(img):
    r = random.randint(0, 3)  #TODO Check and rewrite func
    if r < 1:
        return img.transpose(Image.ROTATE_90)
    elif r < 2:
        return img.transpose(Image.ROTATE_270)
    elif r < 3:
        return img.transpose(Image.ROTATE_180)
    else:
        return img


# Rewrite this function
def random_crop_pairs(crop_size, scale, hr, lr, lr_t=(0, 0)):
    oh_lr = ow_lr = crop_size
    hr_lr_ratio = max(1 / scale, scale)

    oh_hr = ow_hr = oh_lr * hr_lr_ratio
    imw_lr, imh_lr = lr.size
    imw_hr, imh_hr = hr.size
    #TODO put it back
    # assert imh_lr * hr_lr_ratio == imh_hr and imw_lr * hr_lr_ratio == imw_hr, \
    #   'image size of hr and lr don\'t match!'
    if lr_t[0] < 0:
        x0 = lr_t[0]
        x1 = imw_lr - ow_lr + 1
    else:
        x0 = 0
        x1 = imw_lr - ow_lr + 1 - lr_t[0]

    if lr_t[1] < 0:
        y0 = lr_t[1]
        y1 = imh_lr - oh_lr + 1
    else:
        y0 = 0
        y1 = imh_lr - oh_lr + 1 - lr_t[0]

    offy_lr = random.randint(y0, y1)
    offx_lr = random.randint(x0, x1)
    offy_hr, offx_hr = int(offy_lr * hr_lr_ratio), int(offx_lr * hr_lr_ratio)

    return (hr.crop((offx_hr, offy_hr, offx_hr + ow_hr, offy_hr + oh_hr)),
            lr.crop((offx_lr, offy_lr, offx_lr + ow_lr, offy_lr + oh_lr)))


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def tensor2im(image_tensor,
              mean=(0.5, 0.5, 0.5),
              img_mul=2.,
              transpose=True,
              dtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy,
                                (1, 2, 0)) / img_mul + np.array(mean)) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    if not transpose:
        image_numpy = np.transpose(image_numpy, (2, 0, 1))
    return np.around(image_numpy).astype(dtype)


def tensor2ims(tensor, imtype=np.uint8):
    b, c, h, w = tensor.size()
    tensor = tensor.view(-1, h, w)
    tensor = tensor.cpu().float().numpy()
    ims = []
    for i in range(tensor.shape[0]):
        im = tensor[i:i + 1]
        im -= np.min(im)
        crange = np.max(im)
        if not crange == 0:
            im /= crange
            im *= 255
        ims.append(im.transpose((1, 2, 0)).astype(imtype))
    return ims
