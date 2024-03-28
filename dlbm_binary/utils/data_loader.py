"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_binary.
    It contains classes for loading the data.

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

import torch
import numpy as np
import os
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

torch.manual_seed(42)


class SequentialDataset(Dataset):
    """
        Class used for data augmentation
    """
    def __init__(self,
                 dataset_dir='/home/michal/thesis/frames',
                 state_size=8,
                 state_dim=198,
                 flip=False,
                 invert=False,
                 equalize=False,
                 rgb=False):
        """
        :param dataset_dir: directory containing frames
        :param state_size: number of frames in one data sample
        :param state_dim: frame width/height
        :param flip:  boolean information whether image horizontal flip (augmentation technique) should be performed
        :param invert: boolean information whether image invertion (augmentation technique) should be performed
        :param equalize: boolean information whether histogram equalisation (augmentation technique) should be performed
        :param rgb: boolean information whether the input image is RGB (true) or Grayscale (False)
        """

        self.flip = flip
        self.invert = invert
        self.equalize = equalize
        self.aug_type = ""
        self.img_path = None

        self.state_size_data = state_size
        self.state_dim = state_dim
        self.rgb = rgb
        img_labels = self.get_labels(path=dataset_dir)
        self.img_labels, self.img_paths = self.get_img_paths(dataset_dir, img_labels)

        if flip:
            p_horizontal_flip = 1.0
            self.aug_type += 'f'
        else:
            p_horizontal_flip = 0.0

        if invert:
            p_invert = 1.0
            self.aug_type += 'i'
        else:
            p_invert = 0.0

        if equalize:
            p_equalize = 1.0
            self.aug_type += 'e'
        else:
            p_equalize = 0.0

        self.convert = T.Compose([T.ColorJitter(brightness=(0.4, 1.6), contrast=(0.4, 1.6)),
                                  T.RandomHorizontalFlip(p=p_horizontal_flip),
                                  T.RandomInvert(p=p_invert),
                                  T.RandomEqualize(p=p_equalize),
                                  T.Resize((self.state_dim, self.state_dim),
                                           interpolation=T.InterpolationMode.BILINEAR)])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        self.img_path = self.img_paths[idx]
        image_tensor = self.get_tensor_from_image(self.img_paths[idx])

        label = self.img_labels[idx]
        return image_tensor, label

    def get_tensor_from_image(self, paths):
        """
        Loading image as a tensor
        :param paths: paths of the frames
        :return: tensor
        """
        screens = []

        for path in paths:
            screen = Image.open(path)
            screen = np.asarray(screen, dtype=np.uint8)
            screen = torch.tensor(screen).unsqueeze(0)
            screen = self.convert(screen)
            screens.append(screen)

        screen = torch.stack(screens)

        if self.rgb:
            screen = torch.stack([screen, screen, screen]).squeeze()
        return screen

    def get_frame_series(self, path, label_type, image_base):
        """
        Loading paths to the frames series
        :param path: dataset path
        :param label_type: name of the class in the dataset
        :param image_base: base name for a dataset sample
        :return: paths to series of frames
        """
        images = []

        for i in range(self.state_size_data):
            grayfile = os.path.join(path, label_type, f"{image_base}_{i}.jpg")
            images.append(grayfile)

        return images

    def get_labels(self, path):
        """
        Loading labels for data samples
        :param path: path to labels
        :return: list with labels
        """
        return os.listdir(path)

    def get_img_paths(self, path, label_types):
        """
        Loading paths to the frames series
        :param path: path to a directory with frames
        :param label_types: list with names for labels
        :return: labels, paths to frame series
        """
        label_dict = {'diff': 0, 'feedback': 1}
        images = []
        labels = []

        for label_type in label_types:
            img_paths = os.listdir(os.path.join(path, label_type))
            image_bases = ['_'.join(img_path.split('/')[-1].split('_')[:-1]) for img_path in img_paths]
            image_bases = list(set(image_bases))

            for image_base in image_bases:
                if self.state_size_data == 1:
                    images += self.get_frame_series(path, label_type, image_base)
                    labels += [label_dict[label_type]] * self.state_size_data

                else:
                    images.append(self.get_frame_series(path, label_type, image_base))
                    labels.append(label_dict[label_type])

        return np.array(labels), images


class InteractionDataset(Dataset):
    """
        Class used to load data for the training and evaluation purposes
    """
    def __init__(self,
                 path='/home/michal/thesis/interaction_dataset/train',
                 n_frames_data=8,
                 n_frames_used=8,
                 rgb=False,
                 standardization=True,
                 mean=127,
                 std=76):
        """
        :param path: path to the frames
        :param n_frames_data: number of frames in one data sample
        :param n_frames_used: number of frames loaded at once
        :param rgb: boolean information whether the loaded data is in RGB format
        :param standardization: boolean information whether data should be standardized
        :param mean: data mean value for standardization
        :param std: data standard deviation value for standardization
        """
        self.n_frames_data = n_frames_data
        self.n_frames_used = n_frames_used
        self.state_dim = 198
        self.rgb = rgb
        self.standardization = standardization
        self.mean = mean
        self.std = std
        img_labels = self.get_labels(path=path)
        self.img_labels, self.img_paths = self.get_img_paths(path, img_labels)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        self.img_path = self.img_paths[idx]

        if self.n_frames_used == 1:
            image_tensor = self.get_tensor_from_image(self.img_paths[idx])

        else:
            if self.rgb:
                image_tensor = torch.Tensor(self.n_frames_data, 3, self.state_dim, self.state_dim)
            else:
                image_tensor = torch.Tensor(self.n_frames_data, self.state_dim, self.state_dim)

            for i, image_path in enumerate(self.img_paths[idx]):
                image_tensor[i] = self.get_tensor_from_image(image_path)

        label = self.img_labels[idx]
        return image_tensor, label

    def get_tensor_from_image(self, file):
        """
        Loading image as a tensor
        :param file: path to the frame
        :return: tensor
        """
        screen = Image.open(file)

        if self.standardization:
            screen = (np.asarray(screen, dtype=np.float32) - self.mean) / self.std
        else:
            screen = np.asarray(screen, dtype=np.float32) / 255

        screen = torch.tensor(screen).unsqueeze(dim=0)

        if self.rgb:
            screen = torch.stack([screen, screen, screen]).squeeze()
        return screen

    def get_data_paths(self, path, label_type, image_base):
        """
        Loading paths to the frames series
        :param path: path to the directory containing frames
        :param label_type: name of the label
        :param image_base: base name for a frame
        :return: paths to series of frames
        """
        images = []

        for i in range(self.n_frames_data):
            grayfile = os.path.join(path, label_type, f"{image_base}_{i}.jpg")
            images.append(grayfile)

        return images

    def get_labels(self, path):
        """
        Loading labels for data samples
        :param path: path to labels
        :return: list with labels
        """
        return os.listdir(path)

    def get_img_paths(self, path, label_types):
        """
        Loading paths to the frames series
        :param path: path to a directory with frames
        :param label_types: list with names for labels
        :return: labels, paths to frame series
        """
        label_dict = {'diff': 0, 'feedback': 1}
        images = []
        labels = []

        for label_type in label_types:
            img_paths = os.listdir(os.path.join(path, label_type))
            image_bases = ['_'.join(img_path.split('/')[-1].split('_')[:-1]) for img_path in img_paths]
            image_bases = list(set(image_bases))

            for image_base in image_bases:
                if self.n_frames_used == 1:
                    images += self.get_data_paths(path, label_type, image_base)
                    labels += [label_dict[label_type]] * self.n_frames_data

                else:
                    images.append(self.get_data_paths(path, label_type, image_base))
                    labels.append(label_dict[label_type])

        return np.array(labels), images
