"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_binary.
    It is a configuration file for ResNet50 model.

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
# Number of training epochs
epochs = 12

# Network parameters
noutputs = 2
