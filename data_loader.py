import numpy as np
import re
import torch
from torch.utils.data.dataset import Subset
import ipdb


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
    assert training_set.dataset == calibration_set.dataset == noise_eval_set.dataset

    training_indices = training_set.indices
    calibration_indices = calibration_set.indices
    new_clean_indices = noise_eval_set.indices[clean_prediction.bool()]

    all_clean_indices = torch.cat((training_indices, calibration_indices, new_clean_indices), dim=0)

    return Subset(training_set.dataset, all_clean_indices)
