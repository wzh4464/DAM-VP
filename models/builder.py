#!/usr/bin/env python3
"""Model construction functions."""
import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from utils import logging
from models.model_zoo import *

logger = logging.get_logger("dam-vp")
_MODEL_TYPES = {
    "resnet50-1k": ResNet50_1K,
    "vit-b-1k": ViT_B_1K,
    "vit-b-22k": ViT_B_21K,
    "swin-b-1k": Swin_B_1K,
    "swin-b-22k": Swin_B_22K,
    "moco-v3-b-1k": MoCo_v3_B_1K,
    # "clip-resnet50": CLIP_ResNet50,
    # "clip-vit-b": CLIP_ViT_B
}


def _construct_model(args):
    """Build the pretrained model."""
    assert (
        args.pretrained_model in _MODEL_TYPES.keys()
    ), f"Model type '{args.pretrained_model}' not supported"
    if args.device == "cuda":
        assert (
            args.num_gpus <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"

    # Construct the model
    model_type = args.pretrained_model
    model = _MODEL_TYPES[model_type](args)

    model, device = load_model_to_device(model, args)
    logger.info(f"Device used for model: {device}")

    return model, device


def get_current_device():
    return (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )


def load_model_to_device(model, args):
    cur_device = args.local_rank if args.distributed else get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if args.num_gpus > 1:
            # Make model replica operate on the current device
            model = DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=True,
            )
    else:
        model = model.to(cur_device)
    return model, cur_device

class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """A wrapper for DistributedDataParallel."""
    # succeed all the attributes and methods of DistributedDataParallel
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # succeed all the attributes and methods from self.module
        for name in dir(self.module):
            if not hasattr(self, name):
                setattr(self, name, getattr(self.module, name))
