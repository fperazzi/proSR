import os
import os.path as osp
import prosr

import torch
import torchvision.transforms as transforms
import skimage.io as io
import skimage.transform
import numpy as np

from pprint import pprint
from argparse import ArgumentParser

from prosr import config
from prosr.logger import info
from prosr.models import ProSR
from torch.autograd import Variable
import os.path as osp

def parse_args():
  parser = ArgumentParser(description='ProSR')

  parser.add_argument('-w','--weights',type=str,
                      required=True,
                      help='Pretrained model weights.'
                      )

  parser.add_argument('-i','--input',type=str,nargs='+',
                      required=True,
                      help='List of images to upsample')

  parser.add_argument('-s','--upscale-factor',type=int,
                      default=config.defaults.max_scale,
                      help='Upscaling factor.')

  parser.add_argument('-o', '--output-dir',type=str,
                      default=None,
                      help='Output folder.')

  return parser.parse_args()


def make_output_fn(fn,upscale_factor,output_dir=None):
  path,ext = osp.splitext(fn)
  if output_dir is not None:
    path = osp.join(output_dir,osp.basename(path))
  return osp.join('./' + osp.basename(path) + '_proSRx{}'.format(upscale_factor)+ext)

def tensor2im(im_tensor, mean=(0.5, 0.5, 0.5), img_mul=2.,im_residual=None):
  im_numpy = np.transpose(im_tensor[0].cpu().float().numpy(),(1,2,0))
  if im_residual is not None:
    im_numpy += im_residual
  im_numpy = (im_numpy / img_mul + np.array(mean)) * 255.0
  im_numpy = im_numpy.clip(0, 255)
  return np.around(im_numpy).astype(np.uint8)

if __name__ == '__main__':

  # Parse command-line arguments
  args   = parse_args()
  params = config.defaults

  pprint(params)

  net_G = ProSR(params.G,params.max_scale).cuda()
  net_G.load_state_dict(
    torch.load(args.weights))

  net_G.eval()

  # DIV2K dataset statistics
  mean = params.mean_img
  std  = ([1.0/params.mul_img]*3)

  # Set of image transformations applied to the input image
  info('Loading ProSR')
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean,std)
    ])

  # Prepare output folder
  if args.output_dir is not None and \
      not osp.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  info('Processing images:')
  for fn_lr in args.input[:3]:
    lr = io.imread(fn_lr)

    with torch.no_grad():
      lr_t = preprocess(lr)[None,...]
      sr_t = net_G.forward(lr_t.cuda(),args.upscale_factor)

    if params.G.output_residual:
      h,w = sr_t.shape[2:]
      lr_interp = skimage.transform.resize(lr,(h,w),mode='constant')
      lr_interp = (lr_interp - np.array(params.mean_img)) * params.mul_img

    sr = tensor2im(sr_t.data,params.mean_img,params.mul_img,lr_interp)

    fn_sr = make_output_fn(fn_lr,args.upscale_factor,args.output_dir)
    info("Saving results: {}".format(fn_sr))

    io.imsave(fn_sr,sr)
