import torch
from pprint import pprint
from prosr.logger import info


def parse_args():
    parser = ArgumentParser(
        description='Print configuration file of the pretrained model.')
    parser.add_argument(
        'input', help='path to checkpoint', type=str, required=True)

    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    model_data = torch.load(args.input)
    info('Class Name: {}'.format(state_dict['class_name']))
    pprint(state_dict['params'])
