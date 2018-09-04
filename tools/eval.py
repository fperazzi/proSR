from argparse import ArgumentParser
from prosr.logger import error, info
from prosr.metrics import eval_psnr_and_ssim
from prosr.utils import get_filenames, IMG_EXTENSIONS, print_evaluation

import numpy as np
import os.path as osp
import prosr
import skimage.io as io
import torch


def parse_args():
    parser = ArgumentParser(description='Evaluation')
    parser.add_argument(
        '-i',
        '--input',
        help='High-resolution images, either list or path to folder',
        type=str,
        nargs='*',
        required=True,
        default=[])
    parser.add_argument(
        '-t',
        '--target',
        help='Super-resolution images, either list or path to folder',
        type=str,
        nargs='*',
        required=True,
        default=[])
    parser.add_argument(
        '-s',
        '--scale',
        help='upscale ratio e.g. 2, 4 or 8',
        type=int,
        required=True)

    args = parser.parse_args()

    args.input = get_filenames(args.input, IMG_EXTENSIONS)
    args.target = get_filenames(args.target, IMG_EXTENSIONS)

    if not len(args.input):
        error("Did not find images in: {}".format(args.input))

    if len(args.input) != len(args.target):
        error("Inconsistent number of images between 'input' and 'target'")

    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    psnr_mean = 0
    ssim_mean = 0

    for iid, (hr_filename, sr_filename) in enumerate(
            zip(args.target, args.input)):
        hr_img = io.imread(hr_filename)
        sr_img = io.imread(sr_filename)

        psnr_val, ssim_val = eval_psnr_and_ssim(sr_img, hr_img,args.scale)

        print_evaluation(
            osp.basename(sr_filename), psnr_val, ssim_val, iid + 1,
            len(args.target))

        psnr_mean += psnr_val
        ssim_mean += ssim_val

    psnr_mean /= len(args.target)
    ssim_mean /= len(args.target)

    print_evaluation("average", psnr_mean, ssim_mean, None, None)
