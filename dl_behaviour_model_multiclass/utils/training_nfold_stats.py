"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dl_behaviour_model_multiclass.
    It is used for plotting F1 scores and printing performance metrics obtained during n-fold cross-validation.

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
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rc('font', **{'size': 20})
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

USERS = ["1MBU59SJ", "Z7U8NLC9", "U3L9LFS0", "M4OE3RP5", "J0YH72SI",
         "03DEQR10", "Q4GTE6L4", "PTEM0K27", "6XTLNK55", "5J7PWO3G",
         "1PE38CJI", "25NQFBB2", "1CZ1CL1P",
         "6ZN36CQR", "6RGY40ES", "3UDT4XN8", "3G4MPE2W", "76HKXYD3",
         "A9XL9U1N", "COT085MQ", "F41CCF9W", "Q4ABT87L", "SYBO5F61"]
OUTPUT_PATH = "/home/michal/thesis/dl_behaviour_model/plots/nfold"
MODELS = ['dlc1', 'dlc8']
PATH = '/home/michal/thesis/dl_behaviour_model/results/dlc/nfold'


def plot_f1(x, y, path, model='dlc8'):
    fig, ax = plt.subplots()
    plt.figure(figsize=(11.7, 1.5))
    plt.xlabel("Test user")
    plt.ylabel("$F_1$")
    plt.bar(x, y, zorder=2, width=0.5)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.xticks(x)
    plt.xlim((-0.75, len(x) - 0.25))
    plt.grid(axis='y', alpha=0.5, zorder=1)
    plt.savefig(f"{path}/{model}.pdf", bbox_inches="tight")
    #plt.close()


valid_f1 = []
valid_prec = []
valid_recall = []
valid_acc = []

for user in tqdm(USERS):
    checkpoints = os.listdir(os.path.join(PATH, user))
    i = len(checkpoints)-3
    checkpoint = torch.load(os.path.join(PATH, user, f'model_epoch{i}.pt'))
    valid_f1.append(float(checkpoint['f1']))
    valid_prec.append(float(checkpoint['precision']))
    valid_recall.append(float(checkpoint['recall']))
    valid_acc.append(float(checkpoint['accuracy']))

plot_f1(list(range(len(USERS))), valid_f1, OUTPUT_PATH)
print(valid_f1)
print(f'Mean F1: {np.mean(valid_f1)}')
print(f'Mean Precision: {np.mean(valid_prec)}')
print(f'Mean Recall: {np.mean(valid_recall)}')
print(f'Mean Accuracy: {np.mean(valid_acc)}')
