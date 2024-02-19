"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dl_behaviour_model_binary.
    It contains a DLC1/DLC8 neural network class.

    dl_behaviour_model_binary is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    dl_behaviour_model_binary is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with dl_behaviour_model_binary. If not, see <http://www.gnu.org/licenses/>.
"""

import torch.nn as nn

class DLC(nn.Module):
    def __init__(self, noutputs, nfeats, nstates, kernels, strides, poolsize):
        super(DLC, self).__init__()
        self.noutputs = noutputs
        self.nfeats = nfeats
        self.nstates = nstates
        self.kernels = kernels
        self.strides = strides
        self.poolsize = poolsize
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.nfeats, out_channels=self.nstates[0], kernel_size=self.kernels[0],
                      stride=self.strides[0], padding=1),
            nn.BatchNorm2d(nstates[0]),
            nn.ReLU(),
            nn.MaxPool2d(self.poolsize),
            nn.Conv2d(in_channels=self.nstates[0], out_channels=self.nstates[1], kernel_size=self.kernels[1],
                      stride=self.strides[1]),
            nn.BatchNorm2d(self.nstates[1]),
            nn.ReLU(),
            nn.MaxPool2d(self.poolsize),
            nn.Conv2d(in_channels=self.nstates[1], out_channels=self.nstates[2], kernel_size=self.kernels[1],
                      stride=self.strides[1]),
            nn.BatchNorm2d(self.nstates[2]),
            nn.ReLU(),
            nn.MaxPool2d(self.poolsize),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.nstates[2] * self.kernels[1] * self.kernels[1], self.nstates[3]),
            nn.ReLU(),
            nn.Linear(self.nstates[3], self.noutputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.nstates[2] * self.kernels[1] * self.kernels[1])
        x = self.classifier(x)
        return x

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())
