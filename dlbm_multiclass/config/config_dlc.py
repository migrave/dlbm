"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dl_behaviour_model_multiclass.
    It is a configuration file for DLC1/DLC8 model.

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

# Width/height of the input frame
proc_frame_size = 198
# Number of frames in one data sample
state_size_data = 8
# Device to perform calculation
device = "cuda"
# Minibatch size
minibatch_size = 25
# Learning rate
learning_rate = 0.001
# Number of behaviour_models epochs
epochs = 12
# Class labels
labels = ['diff2', 'diff3', 'feedback']

# Network parameters
noutputs = 3
# number of input layers
nfeats = 1  # DLC1
nfeats_full = 8  # DLC8

# Number of kernels for each convolutional layer and number of neurons in the fully connected layer
nstates = [8, 16, 32, 128]  # DLC1
nstates_full = [8, 16, 32, 128]  # DLC8
kernels = [9, 5]
strides = [3, 1]
poolsize = 2
