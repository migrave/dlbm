"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_multiclass.
    It is used to obtain mean and standard deviation values for the dataset.

    dlbm_multiclass is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    dlbm_multiclass is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with dlbm_multiclass. If not, see <http://www.gnu.org/licenses/>.
"""
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

DATASET_PATH = "/home/michal/thesis/interaction_dataset"
SPLIT = "train"
CLASSES = ["diff2", "diff3",  "feedback"]

psum = 0
psum_sq = 0
i = 0

for cat in CLASSES:
    image_paths = os.listdir(os.path.join(DATASET_PATH, SPLIT, cat))
    for image_path in tqdm(image_paths):
        screen = Image.open(os.path.join(DATASET_PATH, SPLIT, cat, image_path))
        screen = np.asarray(screen, dtype=np.float32)
        psum += np.sum(screen)
        psum_sq += np.sum(screen**2)
        i += 1

total_count = 198*198*i
mean = psum / total_count
var = (psum_sq / total_count) - (mean ** 2)
std = np.sqrt(var)

print(mean, std)
# 127.44799950176271 76.25369107706081
# 127.43740033602279 75.90818760716293 - extended dataset