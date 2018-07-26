from __future__ import print_function

import collections
import glob
import os

import numpy as np
from PIL import Image

import torch

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def get_filenames(source, image_format):
    # Seamlessy load single files, list of files and files from directories.
    if len(source) and os.path.isdir(source[0]):
        source_fns = sorted(
            glob.glob("{}/*.{}".format(source[0], image_format)))
        print(source_fns)
    else:
        source_fns = source

    return source_fns


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, mean=(0.5, 0.5, 0.5), img_mul=2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (
        np.transpose(image_numpy, (1, 2, 0)) /
        (1.0 / np.array(img_mul).reshape(1, 1, -1)) + np.array(mean)) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    return np.around(image_numpy).astype(np.uint8)


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


def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    # number of patches, patch_shape
    shape = ((X - x + 1), (Y - y + 1), x, y)
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize * np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def mod_crop(im, scale):
    h, w = im.shape[:2]
    # return im[(h % scale):, (w % scale):, ...]
    return im[:h - (h % scale), :w - (w % scale), ...]


def cut_boundary(im1, scale):
    boundarypixels = 0
    if scale > 1:
        boundarypixels = scale
        im1 = im1[boundarypixels:-boundarypixels, boundarypixels:
                  -boundarypixels, :]

    return im1


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_image(image_numpy, image_path, mode=None):
    image_pil = Image.fromarray(image_numpy, mode).convert('RGB')
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [
        e for e in dir(object)
        if isinstance(getattr(object, e), collections.Callable)
    ]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join([
        "%s %s" % (method.ljust(spacing),
                   processFunc(str(getattr(object, method).__doc__)))
        for method in methodList
    ]))


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print(
            'mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f'
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def spatial_resize(x, size=None, scale_factor=None):
    import torch.nn.functional as F
    # scale_factor has to be integer
    assert (size is not None) or (
        scale_factor is not None), 'must specify scale or scale_factor'
    assert (size is None) or (scale_factor is
                              None), 'cannot specify both size and scale_factor'
    if size is None:
        h, w = x.size()[-2:]
        size = int(h * scale_factor), int(w * scale_factor)
    if h < size[0] and w < size[0]:
        return F.upsample(x, size=size, mode='bilinear')
    else:
        if size[0] % h != 0 or size[1] % w != 0:
            return F.adaptive_avg_pool2d(x, output_size=size)
        else:
            if scale_factor is None:
                assert (size[0] // h) == (size[1] // w), \
                    'scale factor is the same for both dimensions'
                scale_factor = size[0] / h
            return F.avg_pool2d(x, int(1 / scale_factor))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_csv_as_list(csv_file):
    import csv
    try:
        with open(csv_file) as csvfile:
            reader = csv.DictReader(
                csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            datalist = []
            datalist = list(reader)
            return datalist
    except FileNotFoundError as ex:
        raise ex


def truncate_list_of_dict(ld, key, upper_bound):
    """truncate the list of dict according to dict[key] < value, assume the values are sorted"""
    if len(ld) == 0:
        return ld
    for i, d in enumerate(ld):
        if key in d and d[key] >= upper_bound:
            i -= 1
            break
    return ld[:(i + 1)]


def write_list_to_csv(csv_file, data_list, csv_columns=None):
    import csv
    try:
        with open(csv_file, 'w') as csvfile:
            if len(data_list) > 0:
                fieldnames = csv_columns if csv_columns else list(
                    data_list[-1].keys())
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=fieldnames,
                    restval=0.0,
                    extrasaction='ignore',
                    quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                for data in data_list:
                    writer.writerow(data)

    except OSError as ex:
        print("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    return


class Struct:
    '''The recursive class for building and representing objects with.'''

    def __init__(self, obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return '{%s}' % str(', '.join(
            '%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))


def parse_config(defaults_dict, config_dict):
    # recusively update dict with u
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                r = update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        return d

    z = defaults_dict.copy()
    update(z, config_dict)
    return Struct(z)


def print_current_errors(epoch, i, errors, t, log_name=None):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)

    print(message)
    if log_name is not None:
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)


def crop_boundaries(im, cs):
    if cs > 1:
        return im[cs:-cs, cs:-cs, ...]
    else:
        return im
