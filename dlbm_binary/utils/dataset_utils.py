"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_binary.
    It contains functions for dataset augmentation, balancing and splitting.

    dlbm_binary is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    dlbm_binary is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with dlbm_binary. If not, see <http://www.gnu.org/licenses/>.
"""

import os.path
import torch
import numpy as np
from utils.data_loader import SequentialDataset
import itertools
import cv2
from tqdm import tqdm
import copy


def load_dataset(state_size_data):
    """
    Function used to load, augment, balance and split the dataset
    :param state_size_data: number of frames that create one sample
    :return: train, validation and test sets
    """
    datasets = []
    combs = [p for p in itertools.product([True, False], repeat=2)]

    for comb in combs:
        dataset = SequentialDataset(state_size=state_size_data,
                                    invert=comb[0],
                                    flip=comb[1])
        datasets.append(dataset)

    train_set = []
    valid_set = []
    test_set = []

    for dataset in datasets:
        train, valid, test = torch.utils.data.random_split(dataset,
                                                           [0.7, 0.15, 0.15],
                                                           generator=torch.Generator().manual_seed(42))

        train = balance_dataset(train, local_generator=True)
        valid = balance_dataset(valid, local_generator=True)
        test = balance_dataset(test, local_generator=True)

        train_set.append(train)
        valid_set.append(valid)
        test_set.append(test)

    return train_set, valid_set, test_set


def save_dataset(train, valid, test, root_path="../interaction_dataset"):
    """
    Function used to save three splits of the dataset, namely: training, validation and test split
    :param train: behaviour_models dataset split
    :param valid: validation dataset split
    :param test: testing dataset split
    :param root_path: path to save the dataset splits
    :return: None
    """
    for name, dataset in zip(['train', 'valid', 'test'], [train, valid, test]):
        for set in tqdm(dataset):
            for i in range(len(set)):
                imgs, label = set[i]
                imgs = imgs.squeeze().numpy()
                paths = set.img_path

                for i, path in enumerate(paths):
                    file_name = path.split('/')[-1]
                    label = path.split('/')[-2]
                    aug_type = set.aug_type

                    if aug_type:
                        new_path = os.path.join(root_path, name, label, f"{aug_type}_{file_name}")
                    else:
                        new_path = os.path.join(root_path, name, label, f"x_{file_name}")

                    cv2.imwrite(new_path, imgs[i])


def balance_dataset(subset, local_generator=True):
    """
    Function used to balance dataset subset
    :param subset: dataset subset
    :param local_generator: boolean value indicating whether to use a local generator for random sampling
    :return: balanced dataset
    """
    indices = copy.deepcopy(subset.indices)
    dataset = copy.deepcopy(subset.dataset)
    dataset.img_labels = dataset.img_labels[indices]
    dataset.img_paths = [dataset.img_paths[id] for id in indices]

    labels = []
    paths = []

    classes = list(range(0, 2))
    min_size = np.min([len(np.argwhere(dataset.img_labels == action)) for action in classes])

    for id, action in enumerate(classes):
        indices = np.argwhere(dataset.img_labels == action)
        img_labels = dataset.img_labels[indices]
        img_paths = np.array(dataset.img_paths)[indices]

        if local_generator:
            indices = torch.randperm(len(img_labels), generator=torch.Generator().manual_seed(42))[:min_size]
        else:
            indices = torch.randperm(len(img_labels))[:min_size]

        img_labels = img_labels[indices]
        img_paths = img_paths[indices]

        labels = labels + list(img_labels.squeeze())
        paths = paths + [path if isinstance(path, str) else list(path) for path in img_paths.squeeze()]

    dataset.img_labels = np.array(labels)
    dataset.img_paths = paths
    return dataset
