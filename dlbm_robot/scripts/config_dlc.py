"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_robot.
    It is a configuration file for DLC8 model.

    dlbm_robot is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    dlbm_robot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with dlbm_robot. If not, see <http://www.gnu.org/licenses/>.
"""

t_steps = 500

raw_frame_height = 320
raw_frame_width = 240
proc_frame_size = 198
state_size_data = 8
state_size_model = 1
t_episodes = 14
actions = ['1', '2', '3', '4']
device = "cpu"  # cuda
minibatch_size = 25
learning_rate = 0.001
optimizer = "adam"
epochs = 12 #6 for dlc1, 20 for dlc8
labels = ['diff2', 'diff3', 'feedback']

# network
noutputs = 3
nfeats = 1
nfeats_full = 8 # for the model taking 8 frames as an input
nstates = [8, 16, 32, 128] # for input of 1 frame
nstates_full = [8, 16, 32, 128] # for input of 8 frames
# nstates_full = [16, 32, 64, 256] # old, for input of 8 frames
kernels = [9, 5]
strides = [3, 1]
poolsize = 2
