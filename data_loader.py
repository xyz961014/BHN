import numpy as np
import re
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
import ipdb
from PIL import Image


def split_cifar(dataset, clean_ratio, train_ratio):
    indices = torch.randperm(len(dataset))

    clean_size = int(clean_ratio * len(dataset))
    training_size = int(train_ratio * clean_size)
    calibration_size = clean_size - training_size

    training_indices = indices[:training_size]
    calibration_indices = indices[training_size: training_size + calibration_size]
    noise_eval_indices = indices[clean_size:]

    training_set = Subset(dataset, training_indices)
    calibration_set = Subset(dataset, calibration_indices)
    noise_eval_set = Subset(dataset, noise_eval_indices)

    return training_set, calibration_set, noise_eval_set

def parse_ckpt_name(filename):
    """Extract epoch and step numbers from a filename."""
    match = re.search(r'epoch=(\d+)-step=(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1  # Return a default value for non-matching filenames

def merge_clean_dataset(training_set, calibration_set, noise_eval_set, clean_prediction):
    assert training_set.dataset == calibration_set.dataset == noise_eval_set.subset.dataset

    training_indices = training_set.indices
    calibration_indices = calibration_set.indices
    new_clean_indices = noise_eval_set.subset.indices[clean_prediction.bool()]

    all_clean_indices = torch.cat((training_indices, calibration_indices, new_clean_indices), dim=0)

    return Subset(training_set.dataset, all_clean_indices)


class NoisyLabelDataset(Dataset):
    def __init__(self, subset, noisy_labels):
        self.subset = subset
        self.noisy_labels = noisy_labels

    def __getitem__(self, idx):
        data, _ = self.subset[idx]
        noisy_label = self.noisy_labels[idx]
        return data, noisy_label

    def __len__(self):
        return len(self.subset)

    def get_clean_dataset(self, clean_prediction):
        clean_indices = self.subset.indices[clean_prediction.bool()]
        return Subset(self.subset.dataset, clean_indices)


# For Clothing1M, clean part is already divided into training, validation and test
# We precise the subset by giving the clean_X_list.txt file listing the imgs of the subset
class CleanClothing1M(Dataset):
    def __init__(self, subset_list_file, annotation_file, dataset_dir='clothing1M', transform=None, target_transform=None):
        # First, extract the subset from annotation file
        data_dir = Path(dataset_dir)
        annotation_df = pd.read_csv(data_dir / annotation_file, delimiter=' ', header=None)
        subset_files = pd.read_csv(data_dir / subset_list_file, delimiter=' ', header=None)
        subset = list()

        label_dict = {}
        for _, row in tqdm(annotation_df.iterrows(), total=len(annotation_df)):
            label_dict[row[0]] = row[1]
        for _, row in tqdm(subset_files.iterrows(), total=len(subset_files)):
            #query = annotation_df.loc[annotation_df[0] == row[0]]
            file_name = row[0]
            file_label = label_dict[file_name]
            subset.append({'file': file_name, 'label': file_label})
        self.img_labels = pd.DataFrame(subset)
        # Other attributs
        self.img_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        img_path = self.img_dir / self.img_labels.iloc[idx, 0]
        image = Image.open(str(img_path))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.img_labels)


class NoisyClothing1M(Dataset):
    def __init__(self, annotation_file, dataset_dir='clothing1M', transform=None, target_transform=None):
        # First, extract the subset from annotation file
        data_dir = Path(dataset_dir)
        self.img_labels = pd.read_csv(data_dir / annotation_file, delimiter=' ', header=None)
        corrupted_files = [
            "images/6/21/4053357363,404348621.jpg",
                ]
        index_to_drop = []
        for ind, row in tqdm(self.img_labels.iterrows(), total=len(self.img_labels)):
            if row[0] in corrupted_files:
                index_to_drop.append(ind)
        self.img_labels.drop(index_to_drop, inplace=True)
        self.img_labels.reset_index(drop=True, inplace=True)
        # Other attributs
        self.img_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        if type(idx) is torch.Tensor:
            idx = idx.item()
        img_path = self.img_dir / self.img_labels.iloc[idx, 0]
        image = Image.open(str(img_path))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.img_labels)

    def get_clean_dataset(self, clean_prediction):
        clean_indices = torch.tensor(self.img_labels.loc[clean_prediction.bool().tolist()].index)
        return Subset(self, clean_indices)
