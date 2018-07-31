import os.path as osp
from argparse import ArgumentParser

import numpy as np
from skimage import img_as_float
from skimage.color import rgb2ycbcr
from skimage.io import ImageCollection
from skimage.measure import compare_psnr, compare_ssim

from prosr.misc.parallel import Parallel, delayed


def parse_args():

    parser = ArgumentParser()

    parser.add_argument('-i', '--imgs', type=str, nargs=2)
    parser.add_argument('-d', '--dirs', type=str, nargs=2)
    parser.add_argument('-s', '--scale', default=1, type=int)
    parser.add_argument('-m', '--mode', type=str, default='ycbcr')
    parser.add_argument('-p1', '--pattern1', default='*')
    parser.add_argument('-p2', '--pattern2', default='*')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-j', '--n-jobs', type=int, default=1)
    parser.add_argument('--max-images', type=int, default=None)

    return parser.parse_args()


def eval_psnr_and_ssim(im1, im2):
    im1_t = img_as_float(im1)
    im2_t = img_as_float(im2)

    im1_t = rgb2ycbcr(im1_t)[:, :, 0:1] / 255.0
    im2_t = rgb2ycbcr(im2_t)[:, :, 0:1] / 255.0

    if args.scale > 1:
        im1_t = mod_crop(im1_t, args.scale)
        im2_t = mod_crop(im2_t, args.scale)

        im1_t = crop_boundaries(im1_t, int(args.scale)+6)
        im2_t = crop_boundaries(im2_t, int(args.scale)+6)

    psnr_val = compare_psnr(im1_t, im2_t)
    ssim_val = compare_ssim(
        im1_t,
        im2_t,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        K1=0.01,
        K2=0.03,
        sigma=1.5)

    return psnr_val, ssim_val


if __name__ == '__main__':
    args = parse_args()

    if args.imgs is not None:
        imgs1 = ImageCollection(args.imgs[0])
        imgs2 = ImageCollection(args.imgs[1])
    elif args.dirs is not None:
        imgs1 = ImageCollection(args.imgs[0] + "/*")
        imgs2 = ImageCollection(args.imgs[1] + "/*")
    else:
        raise Exception

    # Useful when debugging
    if args.max_images is not None:
        imgs1 = imgs1[:args.max_images]
        imgs2 = imgs2[:args.max_images]

    psnr_val = []
    ssim_val = []

    n_jobs = min(args.n_jobs, len(imgs1))
    print(n_jobs)

    psnr_val, ssim_val = zip(
        *Parallel(n_jobs=n_jobs)(delayed(eval_psnr_and_ssim)(im1, im2)
                                 for im1, im2 in zip(imgs1, imgs2)))

    if args.verbose:
        for i in range(len(imgs1)):
            print('Image: {} | psnr: {:.2f} | ssim: {:.2f}'.format(
                osp.basename(osp.splitext(imgs1.files[i])[0]), psnr_val[i],
                ssim_val[i]))

    mean_psnr = np.average(psnr_val)
    mean_ssim = np.average(ssim_val)

    if args.verbose:
        print('------------------------------------')
    print('Average | psnr: {:.2f} | ssim: {:.2f}'.format(mean_psnr, mean_ssim))
