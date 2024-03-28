"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_binary.
    It is used for plotting training and evaluation loss for all the models.

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
import torch
import numpy as np
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

OUTPUT_PATH = "/home/michal/thesis/dl_behaviour_model/plots/training"
MODELS = ['dlc1', 'dlc8', 'resnet', 'resnet3d']
PATH_DICT = {'dlc1': '/home/michal/thesis/dl_behaviour_model_binary/results/dlc/checkpoints',
             'dlc8': '/home/michal/thesis/dl_behaviour_model_binary/results/dlc/checkpoints_full_state',
             'resnet': '/home/michal/thesis/dl_behaviour_model_binary/results/resnet/checkpoints',
             'resnet3d': '/home/michal/thesis/dl_behaviour_model_binary/results/resnet3d/checkpoints'}


def plot_loss(train_loss_all, valid_loss_all, path, model):
    fig, ax = plt.subplots(1, 1)
    x = np.arange(1, len(train_loss_all) + 1)
    plt.figure(figsize=(6, 3))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, train_loss_all, label='training')
    plt.plot(x, valid_loss_all, label='validation')
    print('Validation loss:', valid_loss_all)
    plt.xticks(x)
    plt.yticks([0.0, 0.2, 0.4, 0.6])
    plt.xlim((0.95, len(train_loss_all) + 0.05))
    plt.legend()
    plt.grid()
    plt.savefig(f"{path}/{model}.pdf", bbox_inches="tight")
    plt.close()


for model in MODELS:
    checkpoints = os.listdir(PATH_DICT[model])

    valid_loss = []
    train_loss = []

    for i in tqdm(range(len(checkpoints))):
        checkpoint = torch.load(os.path.join(PATH_DICT[model], f'model_epoch{i}.pt'))

        valid_loss.append(checkpoint['val_loss_avg'])
        train_loss.append(checkpoint['train_loss_avg'])

    plot_loss(train_loss, valid_loss, OUTPUT_PATH, model)