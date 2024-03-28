"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_multiclass.
    It is used to train DLC1/DLC8.

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

import torch
from behaviour_models.deep_classifier import DeepClassifier

def main():
    torch.cuda.empty_cache()
    torch.manual_seed(torch.initial_seed())

    model_dir = 'results/dlc'

    agent = DeepClassifier(input_state_size=8)

    agent.train(path=model_dir)


if __name__ == "__main__":
    main()
