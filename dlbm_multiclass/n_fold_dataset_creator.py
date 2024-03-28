"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_multiclass.
    It is used to load data and further augment, balance and split it and save as new dataset.

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

from utils.dataset_utils import load_nfold_dataset, save_nfold_dataset

if __name__ == "__main__":
    usr_datasets = load_nfold_dataset(8)
    save_nfold_dataset(usr_datasets)
