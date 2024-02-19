"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dl_behaviour_model_multiclass.
    It is used to augment the frames with five new backgrounds in order to create an extra test set.
    Background images from: https://www.pexels.com/search/office%20background/

    dl_behaviour_model_multiclass is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    dl_behaviour_model_multiclass is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with dl_behaviour_model_multiclass. If not, see <http://www.gnu.org/licenses/>.
"""

import os
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.io import read_image
import torch
import cv2
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm


class Augmentor:
    def __init__(self):
        self.backgrounds_path = "/home/michal/thesis/dl_behaviour_model_multiclass/new_backgrounds/test"
        self.backgrounds = os.listdir(self.backgrounds_path)

        weights = FCN_ResNet50_Weights.DEFAULT
        self.transforms = weights.transforms(resize_size=None)

        self.sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

        self.model = fcn_resnet50(weights=weights, progress=False)
        self.model = self.model.eval()

    def augment(self, image):
        augmented = []
        img = image.repeat(3, 1, 1).unsqueeze(dim=0)

        batch = self.transforms(img)
        output = self.model(batch)['out']

        normalized_mask = torch.nn.functional.softmax(output, dim=1)
        boolean_mask = normalized_mask.argmax(1) == self.sem_class_to_idx['person']

        img = img.squeeze()

        for i, background in enumerate(self.backgrounds):
            background = cv2.imread(os.path.join(self.backgrounds_path, background), cv2.IMREAD_GRAYSCALE)
            background = cv2.resize(background, (198, 198))
            background = torch.from_numpy(background)
            background = background.repeat((3, 1, 1))

            foreground = draw_segmentation_masks(img, masks=~boolean_mask, alpha=1.0)
            background = draw_segmentation_masks(background, masks=boolean_mask, alpha=1.0)

            augmented_img = foreground + background

            augmented.append(augmented_img[0].numpy())

        return augmented


#DATASET_PATH = "/media/michal/migrave/3labels/frames"
DATASET_PATH = "/home/michal/thesis/frames"
OUTPUT_PATH = "/home/michal/thesis/test_frames"

CLASSES = ["diff2", "diff3", "feedback"]

aug = Augmentor()

for cat in CLASSES:
    image_names = os.listdir(os.path.join(DATASET_PATH, cat))

    for image_name in tqdm(image_names):
        img_nm_pts = image_name.split('.')[0].split('_')
        image = read_image(os.path.join(DATASET_PATH, cat, image_name))
        augmented_imgs = aug.augment(image)

        for i, img in enumerate(augmented_imgs):
            new_image_name = '_'.join(img_nm_pts[:-2]) + f'_b{i}_' + '_'.join(img_nm_pts[-2:]) + '.jpg'
            cv2.imwrite(os.path.join(OUTPUT_PATH, cat, f'x_{new_image_name}'), img)