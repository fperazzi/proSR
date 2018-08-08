import os
import os.path as osp
from argparse import ArgumentParser
import yaml
from pprint import pprint
from time import localtime, strftime, time
import sys

from easydict import EasyDict as edict
import numpy as np
import random

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(BASE_DIR, 'lib'))

import prosr
from prosr.data import DataLoader, Dataset
from prosr.logger import info
from prosr.models.trainer import CurriculumLearningTrainer
from prosr.utils import get_filenames, print_current_errors, IMG_EXTENSIONS

CHECKPOINT_DIR = 'data/checkpoints'


def parse_args():
    parser = ArgumentParser(description='training script for ProSR')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-m',
        '--model',
        type=str,
        help='model',
        choices=['prosr', 'prosrs', 'prosrgan'])

    group.add_argument(
        '-c',
        '--config',
        type=str,
        help="Configuration file in 'yaml' format.")

    parser.add_argument(
        '--name',
        type=str,
        help='name of this training experiment',
        default=strftime("%Y-%m-%d-%H:%M:%S", localtime()))
    parser.add_argument(
        '--upscale-factor',
        type=int,
        help='upscale factor',
        default=[2, 4, 8],
        nargs='+')
    parser.add_argument(
        '--start-epoch', type=int, help='start from epoch x', default=0)
    parser.add_argument(
        '--resume',
        type=str,
        help=
        'checkpoint to resume from. E.g. --resume \'best_psnr_x4\' for best_psnr_x4_net_G.pth '
    )
    parser.add_argument(
        '-v',
        '--visdom',
        type=bool,
        help='use visdom to visualize',
        default=False)
    parser.add_argument(
        '-p',
        '--visdom-port',
        type=int,
        help='port used by visdom',
        default=8067)
    parser.add_argument(
        '--use-html', type=bool, help='save log images to html', default=False)

    args = parser.parse_args()

    return args


def set_seed():
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(128)
    else:
        torch.cuda.manual_seed_all(128)
    torch.manual_seed(128)
    random.seed(128)


def main(args):
    set_seed()

    ############### training loader and test loader #################
    train_files = get_filenames(
        args.train.dataset.path, image_format=IMG_EXTENSIONS)
    test_files = get_filenames(
        args.test.dataset.path, image_format=IMG_EXTENSIONS)

    training_dataset = Dataset(
        prosr.Phase.TRAIN, [],
        train_files,
        args.cmd.upscale_factor,
        crop_size=args.train.input_size,
        **args.train.dataset)

    training_data_loader = DataLoader(
        training_dataset, batch_size=args.train.batch_size)

    info('training images = %d' % len(training_data_loader))

    testing_dataset = torch.utils.data.ConcatDataset([
        Dataset(
            prosr.Phase.VAL, [],
            test_files,
            s,
            crop_size=None,
            **args.test.dataset) for s in args.cmd.upscale_factor
    ])
    testing_data_loader = torch.utils.data.DataLoader(testing_dataset)
    info('validation images = %d' % len(testing_data_loader))

    ############# set up trainer ######################
    trainer = CurriculumLearningTrainer(
        args,
        training_data_loader,
        start_epoch=args.cmd.start_epoch,
        save_dir=osp.join(CHECKPOINT_DIR, args.cmd.experiment_id),
        resume_from=args.cmd.resume)
    trainer.set_train()

    log_file = os.path.join(CHECKPOINT_DIR, params.cmd.experiment_id,
                            'loss_log.txt')

    steps_per_epoch = len(trainer.training_dataset)
    total_steps = trainer.start_epoch * steps_per_epoch
    trainer.reset_curriculum_for_dataloader()

    next_eval_epoch = 1
    save_model_freq = 10

    ############# start training ###############
    info('start training from epoch %d, learning rate %e' %
         (trainer.start_epoch, trainer.lr))

    for epoch in range(trainer.start_epoch, args.train.epochs):
        epoch_start_time = time()

        for i, data in enumerate(trainer.training_dataset):
            iter_start_time = time()
            trainer.set_input(**data)
            trainer.forward()
            trainer.optimize_parameters()

            total_steps += 1
            if total_steps % 100 == 0:
                errors = trainer.get_current_errors()
                t = time() - iter_start_time
                print_current_errors(
                    epoch, total_steps, errors, t, log_name=log_file)

        # Save model
        if (epoch + 1) % 1 == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch + 1, total_steps))
            trainer.save(str(epoch + 1))


        ################# update learning rate  #################
        if (epoch - trainer.best_epoch) > args.train.lr_schedule_patience:
            trainer.save('last_lr_%g' % trainer.lr)
            trainer.update_learning_rate()

        ################ visualize ###############
        if args.cmd.visdom:
            print("visdom")
            lrs = {
                'lr%d' % i: param_group['lr']
                for i, param_group in enumerate(
                    trainer.optimizer_G.param_groups)
            }
            visualizer.display_current_results(
                trainer.get_current_test_result(), epoch)
            visualizer.plot(lrs, epoch, 3)
            visualizer.plot(test_result, epoch, 2)

        ################# test with validation set ##############
        if next_eval_epoch == epoch:
            next_eval_epoch = max(next_eval_epoch*2,16)
            with torch.no_grad():
                test_start_time = time()
                # use validation set
                trainer.set_eval()
                trainer.reset_eval_result()
                for i, data in enumerate(testing_data_loader):
                    trainer.set_input(**data)
                    trainer.evaluate()

                t = time() - test_start_time
                test_result = trainer.get_current_eval_result()

                trainer.update_best_eval_result(epoch, test_result)
                info('eval at epoch %d : '%epoch + ' | '.join(
                    ['{}: {:.02f}'.format(k, v) for k, v in test_result.items()]) +
                      ' | time {:d} sec'.format(int(t)))

                info('best at epoch %d : '%trainer.best_epoch + ' | '.join(
                    ['{}: {:.02f}'.format(k, v) for k, v in trainer.best_eval.items()]))

                if trainer.best_epoch == epoch:
                    if len(trainer.best_eval) > 1:
                        best_key = [
                            k for k in trainer.best_eval
                            if trainer.best_eval[k] == test_result[k]
                        ]
                    else:
                        best_key = list(trainer.best_eval.keys())
                    trainer.save('best_' + '_'.join(best_key))

                trainer.set_train()


def change_dict_type(dct, intype, otype):
    dct = otype(dct)
    for k, v in dct.items():
        if isinstance(v, intype):
            dct[k] = change_dict_type(v, intype, otype)
    return dct


if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    if args.config is not None:
        with open(args.config) as stream:
            try:
                params = edict(yaml.load(stream))
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(0)
    else:
        params = edict(getattr(prosr, args.model + '_params'))

    # Add command line arguments
    params.cmd = edict(vars(args))

    params.cmd.experiment_id = '{}_{}'.format(args.model, args.name)
    checkpoint_dir = osp.join(CHECKPOINT_DIR, params.cmd.experiment_id)
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    np.save(osp.join(checkpoint_dir, 'params'), params)

    info('{}'.format(params.cmd.experiment_id))

    if args.visdom:
        from prosr.visualizer import Visualizer
        visualizer = Visualizer(
            params.cmd.experiment_id,
            port=args.visdom_port,
            use_html=args.use_html)

    main(params)
