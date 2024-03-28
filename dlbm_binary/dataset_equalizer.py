"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_binary.
    It is used to guarantee that the validation and test dataset splits have equal number of samples.

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

import os
import numpy as np
import shutil
import random

DATASET_PATH = "/home/michal/thesis/interaction_dataset"
SPLITS = ["train", "valid", "test"]
CLASSES = ["diff", "feedback"]

KEYWORD_DICT = {key: {cat: [] for cat in CLASSES} for key in SPLITS}

for split in SPLITS:
    for cat in CLASSES:
        images = os.listdir(os.path.join(DATASET_PATH, split, cat))
        keys_list = []
        for image in images:
            keys = tuple(image.split('.')[0].split('_')[:-1])
            keys_list.append(keys)
        keys_set = set(keys_list)

        KEYWORD_DICT[split][cat] = keys_set

sizes = []
for split in SPLITS[1:]:
    keys_len = len(KEYWORD_DICT[split]['diff'])
    sizes.append(keys_len)

min_id = np.argmin(sizes)
min_size = sizes[min_id]
res_size = sizes[1-min_id]
to_compensate_num = (res_size - min_size)


for cat in CLASSES:
    keys_to_cut = KEYWORD_DICT['train'][cat]
    samples = random.sample(keys_to_cut, to_compensate_num)
    for sample in samples:
        for f_idx in range(8):
            img_name = '_'.join(sample)+f"_{f_idx}.jpg"
            src_path = os.path.join(DATASET_PATH, 'train', cat, img_name)
            dst_path = os.path.join(DATASET_PATH, SPLITS[1:][min_id], cat, img_name)
            shutil.move(src_path, dst_path)
