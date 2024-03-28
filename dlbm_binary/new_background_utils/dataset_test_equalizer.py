"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_binary.
    It is used to equalize the extra test set (with five new backgrounds or without a background)
    Background images from: https://www.pexels.com/search/office%20background/

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
import random

DATASET_PATH = "/home/michal/thesis/white_background_frames"
CLASSES = ["diff", "feedback"]

KEYWORD_DICT = {cat: [] for cat in CLASSES}

for cat in CLASSES:
    images = os.listdir(os.path.join(DATASET_PATH, cat))
    keys_list = []

    for image in images:
        keys = tuple(image.split('.')[0].split('_')[:-1])
        keys_list.append(keys)
    keys_set = set(keys_list)
    KEYWORD_DICT[cat] = keys_set

sizes = []
for cat in CLASSES:
    keys_len = len(KEYWORD_DICT[cat])
    sizes.append(keys_len)

min_id = np.argmin(sizes)
min_size = sizes[min_id]
res_size = sizes[1-min_id]
to_compensate_num = (res_size - min_size)

keys_to_cut = KEYWORD_DICT[CLASSES[1-min_id]]
samples = random.sample(keys_to_cut, to_compensate_num)

for sample in samples:
    for f_idx in range(8):
        img_name = '_'.join(sample)+f"_{f_idx}.jpg"
        src_path = os.path.join(DATASET_PATH, CLASSES[1-min_id], img_name)
        os.remove(src_path)
