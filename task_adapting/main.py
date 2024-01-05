#!/usr/bin/env python3
from __future__ import print_function
from argparse import Namespace

import os
import sys

import torch
from torch.utils.data import DataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from arguments import Arguments
import utils.logging as logGing
from utils.functional import set_seed
from adapter import Adapter
from data_utils import loader as data_loader
from models import builder as model_builder
from launch import logging_train_setup
from aggregation import averageAggregation, majorityAggregation, nearestAggregation, gaussianAggregation


def load_dataset(args) -> list[DataLoader]:
    """Load datasets for task adaption.
    @ return: a list of dataloaders for meta train, meta val and meta test.
    return[0]: meta train dataloader
    return[1]: meta val dataloader
    return[2]: meta test dataloader
    """
    set_seed(args.seed)
    return [
        data_loader.construct_train_loader(
            args, args.test_dataset),  # torch.utils.data.DataLoader
        data_loader.construct_val_loader(
            args, args.test_dataset
        ),
        data_loader.construct_test_loader(args, args.test_dataset),
    ]


def main():
    """Task adaption on the downstream dataset.
    """
    args, minis_test, metalearner, prompter_path = init()
    # start task adaption
    if args.adapt_method == "prompt_wo_head":
        acc = metalearner.damvp_method(minis_test, prompter_path)

    elif args.adapt_method == "prompt_w_head":
        acc = metalearner.our_method_with_head(minis_test, prompter_path)
        # metalearner.save_checkpoint(os.path.join(BASE_DIR, args.output_dir), minis_test, prompter_path)

    else:
        raise NotImplementedError

    metalearner.logger.info(f"Task adaption accuracy: {acc}")


def init() -> tuple[Namespace, list[DataLoader], Adapter, str]:
    # print transited arguments
    print(" ".join(sys.argv))
    torch.autograd.set_detect_anomaly(True)

    # parse arguments
    args = Arguments(stage='task_adapting').parser().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else 'cpu')

    # setup training env including loggers
    logging_train_setup(args)
    logger = logGing.get_logger("dam-vp")

    # basic configuration
    set_seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")

    # load datasets for meta train or test
    minis_test = load_dataset(args)

    # load pretrained model
    model, cur_device = model_builder._construct_model(args)

    # initialize meta-learner
    metalearner = Adapter(args, model, aggregation_strategy_list=[
            averageAggregation(), 
            majorityAggregation(), 
            nearestAggregation(), 
            gaussianAggregation()
            ]
        )
    metalearner.model.to(cur_device)
    prompter_path = None if args.checkpoint_dir == "" else os.path.join(
        BASE_DIR, args.checkpoint_dir)

    return args, minis_test, metalearner, prompter_path


if __name__ == '__main__':
    # main loop
    main()
