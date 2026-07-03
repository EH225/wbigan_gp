# -*- coding: utf-8 -*-
"""
This module serves as the main driver script for training the WBiGAN models.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from utils import read_yaml
from dataset_utils import get_dataloader
from trainer import Trainer


def run_training(config_name: str):
    """
    This helper function runs training of the model for a given config file specified by
    config_name. This function:
        1. Reads in the config file as a dict
        2. Builds the train and val dataloaders
        3. Constructs the Trainer obj
        4. Run the training loop by calling trainer.train()
    """
    config = read_yaml(os.path.join(CURRENT_DIR, "config", f"{config_name}.yml"))
    dataloaders = {"train": get_dataloader("train", 5),
                   "val": get_dataloader("val", 5)}
    trainer = Trainer(config=config, dataloaders=dataloaders)
    trainer.train()


if __name__ == "__main__":
    pass
    ## do some argparse stuff here
