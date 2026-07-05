# -*- coding: utf-8 -*-
"""
This module serves as the main driver script for training the WBiGAN models.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from utils import read_yaml
import argparse
from dataset_utils import get_dataloader
from trainer import Trainer


def run_pretraining(config_name: str, dataset_dir: str) -> None:
    """
    This helper function runs pre-training of the model for a given config file specified by
    config_name. This function:
        1. Reads in the config file as a dict
        2. Builds the train and val dataloaders
        3. Constructs the Trainer obj
        4. Run the training loop by calling trainer.pretrain()

    :param config_name: The name of the config file to use for running training.
    :param dataset_dir: The directory location of the dataset.
    :return: None.
    """
    config = read_yaml(os.path.join(CURRENT_DIR, "config", f"{config_name}.yml"))
    dataloaders = {"train": get_dataloader(dataset_dir, "train", config["training"].get("batch_size", 128)),
                   "val": get_dataloader(dataset_dir, "val", config["training"].get("batch_size", 128))}
    trainer = Trainer(config=config, dataloaders=dataloaders)
    trainer.pretrain()


def run_training(config_name: str, dataset_dir: str) -> None:
    """
    This helper function runs training of the model for a given config file specified by
    config_name. This function:
        1. Reads in the config file as a dict
        2. Builds the train and val dataloaders
        3. Constructs the Trainer obj
        4. Run the training loop by calling trainer.train()

    :param config_name: The name of the config file to use for running training.
    :param dataset_dir: The directory location of the dataset.
    :return: None.
    """
    config = read_yaml(os.path.join(CURRENT_DIR, "config", f"{config_name}.yml"))
    dataloaders = {"train": get_dataloader(dataset_dir, "train", config["training"].get("batch_size", 128)),
                   "val": get_dataloader(dataset_dir, "val", config["training"].get("batch_size", 128))}
    trainer = Trainer(config=config, dataloaders=dataloaders)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WBi-GAN training loop",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="The name of the config file to be used for training.")
    args = parser.parse_args()
    dataset_dir = os.path.join(CURRENT_DIR, "dataset", "processed")
    run_training(args.config, dataset_dir)  # Run the trailing loop of the WBi-GAN
