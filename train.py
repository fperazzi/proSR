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

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(osp.join(BASE_DIR, 'lib'))

import prosr
from prosr.data import DataLoader, Dataset
from prosr.logger import info
from prosr.models.trainer import CurriculumLearningTrainer
from prosr.utils import get_filenames, print_current_errors, IMG_EXTENSIONS



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

    group.add_argument(
        '-ckpt',
        '--checkpoint',
        type=str,
        help='name of this training experiment',
        )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='name of this training experiment',
        default=None
        )

    parser.add_argument(
        '--upscale-factor',
        type=int,
        help='upscale factor',
        default=[2, 4, 8],
        nargs='+')

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

    if (args.model or args.config) and args.output is None:
        parser.error("--model and --config requires --output.")

    ############# set up trainer ######################
    if args.checkpoint:
        args.output = osp.basename(osp.dirname(args.checkpoint))


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
        input_size=args.data.input_size,
        **args.train.dataset)

    training_data_loader = DataLoader(
        training_dataset, batch_size=args.train.batch_size)

    info('training images = %d' % len(training_data_loader))

    testing_dataset = torch.utils.data.ConcatDataset([
        Dataset(
            prosr.Phase.VAL, [],
            test_files,
            s,
            input_size=None,
            **args.test.dataset) for s in args.cmd.upscale_factor
    ])
    testing_data_loader = torch.utils.data.DataLoader(testing_dataset)
    info('validation images = %d' % len(testing_data_loader))


    trainer = CurriculumLearningTrainer(
        args,
        training_data_loader,
        save_dir=args.cmd.output,
        resume_from=args.cmd.resume)
    trainer.set_train()

    log_file = os.path.join(args.cmd.output,'loss_log.txt')

    steps_per_epoch = len(trainer.training_dataset)
    total_steps = trainer.start_epoch * steps_per_epoch
    trainer.reset_curriculum_for_dataloader()

    next_eval_epoch = 1

    max_eval_frequency = 10
    print_errors_freq  = 100
    save_model_freq    = 100

    ############# start training ###############
    info('start training from epoch %d, learning rate %e' %
         (trainer.start_epoch, trainer.lr))

    for epoch in range(trainer.start_epoch + 1, args.train.epochs + 1):
        iter_start_time = time()
        for i, data in enumerate(trainer.training_dataset):
            trainer.set_input(**data)
            trainer.forward()
            trainer.optimize_parameters()

            total_steps += 1
            if total_steps % print_errors_freq == 0:
                errors = trainer.get_current_errors()
                t = time() - iter_start_time
                iter_start_time = time()
                print_current_errors(
                    epoch, total_steps, errors, t, log_name=log_file)
                break

        # Save model
        if (epoch) % save_model_freq == 0:
            info(
                'saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps),
                bold=True)
            trainer.save(str(epoch),epoch,trainer.lr)

        ################# update learning rate  #################
        if (epoch - trainer.best_epoch) >= 0:
            trainer.save('last_lr_%g' % trainer.lr,epoch,trainer.lr)
            trainer.update_learning_rate()

        ################ visualize ###############

        # if args.cmd.visdom:
        #     lrs = {
        #         'lr%d' % i: param_group['lr']
        #         for i, param_group in enumerate(
        #             trainer.optimizer_G.param_groups)
        #     }
        #     visualizer.display_current_results(
        #         trainer.get_current_test_result(), epoch)
        #     visualizer.plot(lrs, epoch, 3)
        #     visualizer.plot(test_result, epoch, 2)

        ################# test with validation set ##############
        if next_eval_epoch == epoch:
            next_eval_epoch = min(next_eval_epoch * 2, max_eval_frequency)
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
                info(
                    'eval at epoch %d : ' % epoch + ' | '.join([
                        '{}: {:.02f}'.format(k, v)
                        for k, v in test_result.items()
                    ]) + ' | time {:d} sec'.format(int(t)),
                    bold=True)

                info(
                    'best at epoch %d : ' % trainer.best_epoch + ' | '.join([
                        '{}: {:.02f}'.format(k, v)
                        for k, v in trainer.best_eval.items()
                    ]),
                    bold=True)

                if trainer.best_epoch == epoch:
                    if len(trainer.best_eval) > 1:
                        best_key = [
                            k for k in trainer.best_eval
                            if trainer.best_eval[k] == test_result[k]
                        ]
                    else:
                        best_key = list(trainer.best_eval.keys())
                    trainer.save('best_' + '_'.join(best_key),epoch,trainer.lr)

            trainer.set_train()


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
    elif args.model is not None:
        params = edict(getattr(prosr, args.model + '_params'))

    else:
        params = torch.load(args.checkpoint + '_net_G.pth')['params']

    # Add command line arguments
    params.cmd = edict(vars(args))
    pprint(params)

    if not osp.isdir(args.output):
        os.makedirs(args.output)
    np.save(osp.join(args.output, 'params'), params)

    experiment_id = osp.basename(args.output)

    info('experiment ID: {}'.format(experiment_id))

    # if args.visdom:
    #     from prosr.visualizer import Visualizer
    #     visualizer = Visualizer(
    #         experiment_id,
    #         port=args.visdom_port,
    #         use_html=args.use_html)

    main(params)
