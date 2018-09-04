from argparse import ArgumentParser
import torch
from pprint import pprint
from prosr.logger import info


def parse_args():
    parser = ArgumentParser(
        description='Print configuration file of the pretrained model.')
    parser.add_argument(
        'input', help='path to checkpoint', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    params_dict = torch.load(args.input)
    info('Class Name: {}'.format(params_dict['class_name']))
    pprint(params_dict['params'])
