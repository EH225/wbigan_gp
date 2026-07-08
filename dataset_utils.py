# -*- coding: utf-8 -*-
"""
This module contains functions for pre-processing the data set into [128, 128, 3] images and constructs
the data loaders required for training.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import Resize, CenterCrop, InterpolationMode
from torchvision import transforms
from utils import get_device
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


def resize_and_crop(input_dir: str, output_dir: str, size: int) -> None:
    """
    Resizes and center-crops all images found in input_dir and saves a new copy to output_dir
    with the same name. This is used as a dataset pre-processing step to create images that are
    all the same size.

    :param input_dir: A directory of input images to read from.
    :param output_dir: The directory where processed images are to be saved.
    :param size: The size to create i.e. (size, size, 3) for each image.
    :return: None. Files are read and saved to disk.
    """
    os.makedirs(output_dir, exist_ok=True)  # Make sure the output_dir exists, create if needed
    # Define image transforms to apply to each image
    resize = Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    center_crop = CenterCrop(size)

    valid_exts = {"jpg", "jpeg", "png"}

    for img_name in tqdm(os.listdir(input_dir), ncols=75):  # Loop over all images from input_dir
        if img_name.split(".")[-1].lower() not in valid_exts:  # Check that this is an image file
            continue

        with Image.open(os.path.join(input_dir, img_name)) as img:  # Open the image
            img = img.convert("RGB")
            img = resize(img)  # Resize it to be a min(H, W) = size
            img = center_crop(img)  # Center crop so that we have [size, size, 3] images
            # Save the image to the output directory when finished processing
            img.save(os.path.join(output_dir, img_name))


class OxfordPetsDataset(Dataset):
    """
    Dataset used to train the WBiGAN on the Oxford Pets dataset.
    """

    def __init__(self, img_dir: str, meta_path: str, transform=None, split: str = "train"):
        """
        Initializes the dataloader for a given image directory and meta-data df path.

        :param img_dir: A file path to the directory containing the images to be loaded.
        :param meta_path: A file path specifying the location of the meta_data.csv file.
        :param transform: A set of transform compositions from albumentations to apply to each
            (image, labels) pair that is loaded.
        :param: The dataset split to use in this dataset.
        """
        self.img_dir = img_dir  # Images will be loaded from this directory for each batch
        self.meta_df = pd.read_csv(meta_path)  # Load in the full df from disk
        self.meta_df = self.meta_df.loc[self.meta_df["split"] == split, :]  # Subset for the split
        self.transform = transform
        self.image_names = self.meta_df["image_name"].tolist()
        # Create a dictionary of class_id value [0, 36] for every image name for quick retrival
        self.class_id = {row["image_name"]: row["class_id"] - 1 for _, row in self.meta_df.iterrows()}

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing keys "image" and "class_id" for a particular index in the dataset.

        :param idx: An internal image index number from [0, len(self.img_ids) - 1].
        :returns: A dict with the following format:
            image: An image of size (C, H, W) as a torch.Tensor
            class_id: The class_id associated with the image, a unique group identifier.
        """
        # Load in the image and labels from disk
        image_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, f"{image_name}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:  # Apply transforms if specified
            image = self.transform(image)

        class_id = torch.tensor(self.class_id[image_name], dtype=torch.long)
        return {"image": image, "class_id": class_id}


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Add random horizontal flip data augmentations
    transforms.ToTensor(),  # Convert to a tensor, [0, 255] -> [0.0, 1.0] pixel values
    transforms.Normalize(
        # Normalize from [0, 1] to [-1, 1] pixel values to match the generator
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to a tensor, [0, 255] -> [0.0, 1.0] pixel values
    transforms.Normalize(
        # Normalize from [0, 1] to [-1, 1] pixel values to match the generator
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])


def get_dataloader(dataset_dir: str, split: str, batch_size: int) -> DataLoader:
    """
    Returns a Dataloader for the split specified i.e. train or val.

    :param dataset_dir: The location of the dataset on disk, which should be a directory
        containing an images folder and a meta_data.csv file.
    :param split: The dataset split to construct a dataloader for i.e. train or val.
    :param batch_size: The number of images in each batch.
    :return: A dataloader for the given split specified in the input args.
    """
    img_dir = os.path.join(dataset_dir, "images")
    meta_path = os.path.join(dataset_dir, "meta_data.csv")

    device = get_device()  # Auto-detect what hardware is available
    if device == "cuda":  # Change kwargs depending on the device in use
        num_workers, pin_memory, persistent_workers, prefetch_factor = 8, True, True, 4
    else:
        num_workers, pin_memory, persistent_workers, prefetch_factor = 0, False, False, None

    if split == "train":
        transform = train_transform
        shuffle = True
    elif split == "val":
        transform = val_transform
        shuffle = False
    else:
        raise KeyError(f"split={split} not recognized")

    dataset = OxfordPetsDataset(img_dir, meta_path, transform, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=persistent_workers,
                      prefetch_factor=prefetch_factor)


if __name__ == "__main__":
    # Pre-process all the images on disk to be [128, 128, 3]
    print("Pre-processing images to be [128, 128, 3]")
    input_dir = os.path.join(CURRENT_DIR, "dataset", "oxford_pets", "original", "images")
    output_dir = os.path.join(CURRENT_DIR, "dataset", "oxford_pets", "processed", "images")
    resize_and_crop(input_dir, output_dir, 128)
