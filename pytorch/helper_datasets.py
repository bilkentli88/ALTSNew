"""
helper_datasets.py

This module contains functions for loading datasets and converting data into bags.
"""

import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

DATASET_MAIN_FOLDER = "Datasets"

def load_dataset(dataset_name):
    """
    Loads the dataset using the load_data function.

    Parameters:
    - dataset_name (str): The name of the dataset.

    Returns:
    - tuple: Train features, train labels, test features, test labels.
    """
    return load_data(DATASET_MAIN_FOLDER, dataset_name)

def load_data(folder, dataset, extension=""):
    """
    Loads train and test data for the given dataset.

    Parameters:
    - folder (str): The folder containing the dataset files.
    - dataset (str): The name of the dataset.
    - extension (str, optional): The file extension (default is "").

    Returns:
    - tuple: Train features, train labels, test features, test labels.
    """
    train_file_base = "".join([dataset, "_TRAIN"])
    test_file_base = "".join([dataset, "_TEST"])

    # Attempt to load files without extension
    train_file = os.path.join(folder, train_file_base + extension)
    test_file = os.path.join(folder, test_file_base + extension)

    # Try loading without extension, if fails, try with .txt extension
    try:
        train_data = np.loadtxt(train_file)
    except FileNotFoundError:
        train_file = os.path.join(folder, train_file_base + ".txt")
        train_data = np.loadtxt(train_file)

    try:
        test_data = np.loadtxt(test_file)
    except FileNotFoundError:
        test_file = os.path.join(folder, test_file_base + ".txt")
        test_data = np.loadtxt(test_file)

    # Normalize features
    train_features = (train_data[:, 1:] - train_data[:, 1:].mean(axis=1).reshape(-1, 1)) / train_data[:, 1:].std(axis=1).reshape(-1, 1)
    test_features = (test_data[:, 1:] - test_data[:, 1:].mean(axis=1).reshape(-1, 1)) / test_data[:, 1:].std(axis=1).reshape(-1, 1)

    # One-hot encode labels
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_data[:, 0].reshape(-1, 1))

    train_labels = enc.transform(train_data[:, 0].reshape(-1, 1)).toarray()
    test_labels = enc.transform(test_data[:, 0].reshape(-1, 1)).toarray()

    return train_features, train_labels.astype(np.int32), test_features, test_labels.astype(np.int32)

def convert_to_bags(data, bag_size, stride_ratio):
    """
    Converts the input data into bags of a specified size with a given stride ratio.

    Parameters:
    - data (np.ndarray or torch.Tensor): Input data.
    - bag_size (int): Size of the shapelet bag.
    - stride_ratio (float): Stride ratio for the sliding window.

    Returns:
    - torch.Tensor: The bags as a tensor.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    bag_size = int(bag_size)
    bags = []
    stride = int(max(round(stride_ratio * bag_size), 1))
    for i in range(data.shape[0]):
        instance = []
        size = data[i].shape[0]
        window = int(bag_size)
        while True:
            instance.append(data[i][window - bag_size: window])
            window += stride
            if window >= size:
                window = size
                instance.append(data[i][window - bag_size: window])
                break
        bags.append(np.array(instance))
    return torch.from_numpy(np.array(bags)).float()

def get_bag_size(dataset_name, bag_ratio):
    """
    Computes the bag size based on a ratio of the time series length.

    Parameters:
    - dataset_name (str): The name of the dataset.
    - bag_ratio (float): The ratio to compute the bag size.

    Returns:
    - int: The computed bag size.
    """
    train_file = "".join([dataset_name, "_TRAIN", ""])
    train_file = os.path.join(DATASET_MAIN_FOLDER, train_file)
    train_data = np.loadtxt(train_file)
    time_series_size = train_data.shape[1] - 1
    bag_size = int(time_series_size * bag_ratio)
    return bag_size
