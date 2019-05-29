import argparse
from datetime import datetime

from GDGAN import gdgan_config


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=7412)

"""
for logs
"""
parser.add_argument('--test_every', type=int, default=5)
parser.add_argument('--save_every', type=int, default=5)

"""
subparsers for each model
"""
subparsers = parser.add_subparsers(dest='model')
gdgan_config.create_subparser(subparsers)


def validate_args(args):
    args.date_str = datetime.now().strftime('%Y_%m_%d_%H_%M')
    return args


config = validate_args(parser.parse_args())
