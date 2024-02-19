'''
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dl_behaviour_model_binary.
    It is used to train ResNet50.

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
'''


import torch
from behaviour_models.pretrained import DeepClassifier
from config import config_resnet as dcfg


def main():
    torch.cuda.empty_cache()
    torch.manual_seed(torch.initial_seed())

    model_dir = 'results/resnet'

    agent = DeepClassifier(cfg=dcfg, input_state_size=1, rgb=True, name="resnet")

    agent.train(path=model_dir)


if __name__ == "__main__":
    main()
