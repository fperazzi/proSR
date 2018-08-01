import os
import os.path as osp
from argparse import ArgumentParser
from pprint import pprint
from time import localtime, strftime, time
import sys

import numpy as np
import random

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(BASE_DIR, 'lib'))

import prosr
from prosr.data import DataLoader, Dataset
from prosr.logger import info
from prosr.models.trainer import CurriculumLearningTrainer
from prosr.utils import get_filenames, print_current_errors

def print_evaluation(filename, psnr, ssim):
    print('{} | psnr: {:.2f} | ssim: {:.2f}'.format(filename, psnr, ssim))


def parse_args():
    parser = ArgumentParser(description='training script for ProSR')
    parser.add_argument('-m', '--model', type=str, help='model', choices=['prosr', 'prosrs', 'prosrgan', 'edsr'], default='prosrl')
    parser.add_argument('--name',        type=str, help='name of this training experiment', default=strftime("%Y-%m-%d-%H:%M:%S", localtime()))
    parser.add_argument('--upscale-factor', type=int, help='upscale factor', default=[2, 4, 8], nargs='+')
    parser.add_argument('--start-epoch', type=int, help='start from epoch x', default=0)
    parser.add_argument('--resume',      type=str, help='checkpoint to resume from')
    parser.add_argument('--eval-dataset', type=str, help='dataset for evaluation', default='Set14')
    parser.add_argument('-v', '--visdom', type=bool, help='use visdom to visualize', default=True)
    parser.add_argument('-p', '--visdom-port', type=int, help='port used by visdom', default=8067)
    parser.add_argument('--use-html', type=bool, help='save log images to html', default=False)
    # parser.add_argument('-b', '--batch-size', type=int, help='batch size', default=16)
    args = parser.parse_args()
    return args

def set_seed():
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(128)
    else:
        torch.cuda.manual_seed_all(128)
    torch.manual_seed(128)
    random.seed(128)


def main(opt):
    set_seed()

    ############### training loader and eval loader #################
    train_files = get_filenames('data/datasets/DIV2K/DIV2K_train_HR', image_format='png')
    eval_files  = get_filenames('data/datasets/{}/HR'.format(args.eval_dataset), image_format='png')
    training_dataset = Dataset(prosr.Phase.TRAIN, [], train_files, args.upscale_factor, crop_size=opt.train.input_size,
                      **opt.data)
    training_data_loader = DataLoader(training_dataset, batch_size=opt.train.batch_size)
    print('#training images = %d' % len(training_data_loader))

    testing_dataset = torch.utils.data.ConcatDataset([Dataset(prosr.Phase.VAL, [], eval_files, s, crop_size=None, **opt.data) for s in args.upscale_factor])
    testing_data_loader = torch.utils.data.DataLoader(testing_dataset)
    print('#validation images = %d' % len(testing_data_loader))

    ############# set up trainer ######################
    trainer = CurriculumLearningTrainer(opt, training_data_loader,
        start_epoch=args.start_epoch, save_dir=osp.join('data', 'checkpoints', opt.experiment_id), resume_from=None)
    trainer.set_train()

    log_file = os.path.join('data/checkpoints', params.experiment_id, 'loss_log.txt')

    steps_per_epoch = len(trainer.training_dataset)
    total_steps = trainer.start_epoch * steps_per_epoch
    trainer.reset_curriculum_for_dataloader()

    ############# start training ###############
    info('start training from epoch %d, learning rate %e' % (trainer.start_epoch, trainer.lr))

    for epoch in range(trainer.start_epoch, opt.train.epochs):
        epoch_start_time = time()
        iter_start_time = time()
        for i, data in enumerate(trainer.training_dataset):
            trainer.set_input(**data)
            trainer.forward()
            trainer.optimize_parameters()

            total_steps += 1
            if total_steps % 100 == 0:
                errors = trainer.get_current_errors()
                t = time() - iter_start_time
                print_current_errors(epoch, total_steps, errors, t, log_name=log_file)

        if (epoch+1) % 10 == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch+1, total_steps))
            trainer.save(str(epoch+1))

        ################# evaluation with validation set ##############
        with torch.no_grad():
            eval_start_time = time()
            # use validation set
            trainer.set_eval()
            trainer.reset_eval_result()
            for i, data in enumerate(testing_data_loader):
                trainer.set_input(**data)
                trainer.evaluate()

            t = time() - eval_start_time
            eval_result = trainer.get_current_eval_result()

            trainer.update_best_eval_result(epoch, eval_result)
            print('evaluation on ' + args.eval_dataset +
                 ', ' + ' | '.join(['{}: {:.02f}'.format(k, v) for k, v in eval_result.items()]) +
                 ', time {:d} sec'.format(int(t)))
            info('best so far in epoch %d: ' % trainer.best_epoch +
                ', '.join(['%s = %.2f' % (k, v) for (k, v) in trainer.best_eval.items()]))

            if trainer.best_epoch == epoch:
                if len(trainer.best_eval) > 1:
                    best_key = [k for k in trainer.best_eval if trainer.best_eval[k] == eval_result[k]]
                else:
                    best_key = list(trainer.best_eval.keys())
                trainer.save('best_'+'_'.join(best_key))

            trainer.set_train()

        ################# update learning rate  #################
        if (epoch - trainer.best_epoch) > opt.train.lr_schedule_patience:
            trainer.save('lastlr_%g' % trainer.lr)
            trainer.update_learning_rate()

        ################ visualize ###############
        if args.visdom:
            lrs = {'lr%d' % i: param_group['lr']
                for i, param_group in enumerate(trainer.optimizer_G.param_groups)}
            visualizer.display_current_results(trainer.get_current_eval_result(), epoch)
            visualizer.plot(lrs, epoch, 3)
            visualizer.plot(eval_result, epoch, 2)


if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    params = getattr(prosr, args.model+'_params')
    pprint(params)

    params.experiment_id = args.model+args.name
    checkpoint_dir = osp.join('data/checkpoints', params.experiment_id)
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    np.save(osp.join(checkpoint_dir, 'params'), params)

    info('{}'.format(params.experiment_id))

    if args.visdom:
        from prosr.visualizer import Visualizer
        visualizer = Visualizer(params.experiment_id, port=args.visdom_port, use_html=args.use_html)

    main(params)
