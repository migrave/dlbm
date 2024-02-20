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

proc_frame_size = 198
state_size_data = 8
device = "cpu"  # cuda

# network
noutputs = 3
nfeats_full = 8
nstates_full = [8, 16, 32, 128]
kernels = [9, 5]
strides = [3, 1]
poolsize = 2
