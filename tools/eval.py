import os.path as osp
from argparse import ArgumentParser

import numpy as np
import skimage.io as io
import torch

import prosr
from prosr.logger import error, info
from prosr.metrics import eval_psnr_and_ssim
from prosr.utils import get_filenames


def print_evaluation(filename, psnr, ssim, iid, n_images):
    print('[{:03d}/{:03d}] {} | psnr: {:.2f} | ssim: {:.2f}'.format(
        iid, n_images, filename, psnr, ssim))


def parse_args():
    parser = ArgumentParser(description='Evaluation')
    parser.add_argument(
        '-hr',
        '--hr-input',
        help='High-resolution images, either list or path to folder',
        type=str,
        nargs='*',
        required=False,
        default=[])
    parser.add_argument(
        '-sr',
        '--sr-input',
        help='Super-resolution images, either list or path to folder',
        type=str,
        nargs='*',
        required=False,
        default=[])
    parser.add_argument(
        '-f', '--fmt', help='Image file format', type=str, default='*')
    parser.add_argument(
        '-u',
        '--upscale-factor',
        help='upscale ratio e.g. 2, 4 or 8',
        type=int,
        required=True)

    args = parser.parse_args()

    args.sr_input = get_filenames(args.sr_input, args.fmt)
    args.hr_input = get_filenames(args.hr_input, args.fmt)

    if not len(args.sr_input):
        error("Did not find images in: {}".format(args.sr_input))

    if len(args.sr_input) != len(args.hr_input):
        error(
            "Inconsistent number of images between 'sr_input' and 'hr_input'")

    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    psrn_mean = 0
    ssim_mean = 0

    for iid, (hr_filename, sr_filename) in enumerate(
            zip(args.hr_input, args.sr_input)):
        hr_img = io.imread(hr_filename)
        sr_img = io.imread(sr_filename)

        psnr_val, ssim_val = eval_psnr_and_ssim(sr_img, hr_img,
                                                args.upscale_factor)

        print_evaluation(
            osp.basename(sr_filename), psnr_val, ssim_val, iid + 1,
            len(args.hr_input))

        psrn_mean += psnr_val
        ssim_mean += ssim_val

    psnr_mean /= len(args.hr_input)
    ssim_mean /= len(args.hr_input)

    print_evaluation("average", psnr_mean, ssim_mean)
