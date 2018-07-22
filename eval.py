import numpy as np
from skimage import img_as_float
from skimage.measure import compare_ssim, compare_psnr
from skimage.color import rgb2ycbcr
import subprocess
import os
import tempfile
import shutil

from argparse import ArgumentParser
from os import listdir
from os.path import join, isfile
from skimage.io import imread, imsave

def mod_crop(im, scale):
  h, w = im.shape[:2]
  return im[:h-(h % scale), :w-(w % scale), ...]

def crop_boundaries(im, cs):
  if cs > 1:
    return im[cs:-cs, cs:-cs, ...]
  else:
    return im

def parse_args():

  parser = ArgumentParser()

  parser.add_argument('-i','--imgs',type=str, nargs=2)
  parser.add_argument('-d','--dirs',type=str, nargs=2)
  parser.add_argument('-s', '--scale',default=1,type=int)
  parser.add_argument('-m', '--mode', type=str, default='ycbcr')
  parser.add_argument('--no-psnr', action='store_false')
  parser.add_argument('--no-ssim', action='store_false')
  parser.add_argument('-p1','--pattern1', default='*')
  parser.add_argument('-p2','--pattern2', default='*')
  parser.add_argument('-v','--verbose', action='store_true')

  return parser.parse_args()


import os.path as osp
import skimage
from skimage.io import ImageCollection
from skimage import img_as_float

if __name__ == '__main__':
  args = parse_args()

  imgs1 = ImageCollection(args.imgs[0]+"/*")
  imgs2 = ImageCollection(args.imgs[1]+"/*")

  psnr_val = []
  ssim_val = []

  for idx,(im1,im2) in enumerate(zip(imgs1,imgs2)):
    im1_t = img_as_float(im1)
    im2_t = img_as_float(im2)

    im1_t = rgb2ycbcr(im1_t)[:, :, 0:1]/255.0
    im2_t = rgb2ycbcr(im2_t)[:, :, 0:1]/255.0

    if args.scale > 1:
      im1_t = mod_crop(im1_t, args.scale)
      im2_t = mod_crop(im2_t, args.scale)

      im1_t = crop_boundaries(im1_t, int(args.scale))
      im2_t = crop_boundaries(im2_t, int(args.scale))

    psnr_val.append(compare_psnr(im1_t, im2_t))
    ssim_val.append(compare_ssim(im1_t, im2_t,
                        win_size=11,
                        gaussian_weights=True,
                        multichannel=True,
                        K1=0.01,
                        K2=0.03,
                        sigma=1.5))
    if args.verbose:
      print('Image: {} | psnr: {:.2f} | ssim: {:.2f}'.format(
        osp.basename(osp.splitext(imgs1.files[idx])[0]),psnr_val[-1],ssim_val[-1]))

  mean_ssim = np.average(psnr_val)
  mean_psnr = np.average(ssim_val)

  if args.verbose:
    print '------------------------------------'
  print('Average | psnr: {:.2f} | ssim: {:.2f}'.format(
    mean_psnr, mean_ssim)

  print(ssim_val,psnr_val)
