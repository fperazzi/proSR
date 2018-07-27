import os
import os.path as osp
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import skimage.io as io
import skimage.transform

import prosr
import torch
import torchvision.transforms as transforms
from prosr import config
from prosr.data import DataLoader, Dataset
from prosr.logger import info
from prosr.metrics import eval_psnr_and_ssim
from prosr.models import EDSR, ProSR
from prosr.utils import get_filenames, tensor2im


def print_evaluation(filename, psnr, ssim):
    print('{} | psnr: {:.2f} | ssim: {:.2f}'.format(filename, psnr, ssim))


def parse_args():
    parser = ArgumentParser(description='ProSR')

    parser.add_argument(
        '-c', '--checkpoint', type=str, required=True, help='Checkpoint')

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Input images, either list or path to folder')

    parser.add_argument(
        '-t',
        '--target',
        type=str,
        nargs='+',
        required=False,
        default=[],
        help='Target images, either list or path to folder')

    parser.add_argument(
        '-u',
        '--upscale-factor',
        type=int,
        required=True,
        help='List of images to upsample')

    parser.add_argument(
        '-f', '--fmt', type=str, default='*', help='Image file format')

    parser.add_argument(
        '-o', '--output-dir', type=str, default='./', help='Output folder.')

    args = parser.parse_args()

    args.input = get_filenames(args.input, args.fmt)
    args.target = get_filenames(args.target, args.fmt)

    return args


if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()
    params = config.default_params

    info()
    pprint(params)

    checkpoint = torch.load(args.checkpoint)
    cls_model = getattr(prosr.models, checkpoint['class_name'])

    model = cls_model(**checkpoint['params']['G'])
    model.load_state_dict(checkpoint['state_dict'])

    info('Phase: {}'.format(config.phase.TEST))
    info('Checkpoint: {}'.format(osp.basename(args.checkpoint)))

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # TODO Change
    dataset = Dataset(config.phase.TEST, args.input, args.target,
                      **checkpoint['params']['data'])

    data_loader = DataLoader(dataset, batch_size=1)

    mean = checkpoint['params']['data']['mean']
    stddev = checkpoint['params']['data']['stddev']

    with torch.no_grad():
        for data in data_loader:
            output = model(data['input'].cuda(), args.upscale_factor)
            sr_img = tensor2im(output, mean, stddev)
            lr_img = tensor2im(data['input'], mean, stddev)
            if 'target' in data:
                hr_img = tensor2im(data['target'], mean, stddev)
                print_evaluation(
                    osp.basename(data['input_fn'][0]),
                    *eval_psnr_and_ssim(sr_img, hr_img, args.upscale_factor))

            io.imsave(
                osp.join(args.output_dir, osp.basename(data['input_fn'][0])),
                sr_img)
