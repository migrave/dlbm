"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dl_behaviour_model_binary.
    It is used to test DLC1 and DLC8 on the extra test sets (with five new backgrounds/without background).

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

import torch
from behaviour_models.deep_classifier import DeepClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt
import torchmetrics
from utils.data_loader import InteractionDataset

matplotlib.rc('font', **{'size': 24})
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

torch.manual_seed(42)

agent_full = DeepClassifier(input_state_size=8)
agent = DeepClassifier(input_state_size=1)

test = InteractionDataset(path='/home/michal/thesis/white_background_frames', n_frames_used=8)
#test = InteractionDataset(path='/home/michal/thesis/test_frames', n_frames_used=8)

testloader = torch.utils.data.DataLoader(test,
                                         batch_size=1,
                                         shuffle=True)

name_suffix = "new_background"

predicted_majority = []
predicted_full = []
ground_truth = []

outputs = 2
display_labels = ['diff', 'feedback']

for images, label in testloader:
    ground_truth.append(label.clone().tolist()[0])

    pred_full = agent_full.predict(images)
    predicted_full.append(pred_full.tolist()[0])

    pred_majority = agent.majority_vote(images)
    predicted_majority.append(pred_majority)

accuracy = torchmetrics.classification.Accuracy(task='binary')
precision = torchmetrics.classification.Precision(task='binary')
recall = torchmetrics.classification.Recall(task='binary')
f1_score = torchmetrics.classification.F1Score(task='binary')
conf_mat = torchmetrics.classification.ConfusionMatrix(task='binary')

gt_labels = torch.FloatTensor(ground_truth)

for predicted, name in zip([predicted_full, predicted_majority],
                           ['DLC8', 'DLC1']):
    prediction_labels = torch.FloatTensor(predicted)

    accuracy_score = accuracy(prediction_labels, gt_labels)
    recall_score = recall(prediction_labels, gt_labels)
    precision_score = precision(prediction_labels, gt_labels)
    f1score = f1_score(prediction_labels, gt_labels)

    print(f'{name} -- Accuracy: {accuracy_score:.3f}, '
          f'precision: {precision_score:.3f}, '
          f'recall: {recall_score:.3f}, '
          f'f1 score: {f1score:.3f}')

conf_matrix_full = confusion_matrix(ground_truth, predicted_full)
conf_matrix_majority = confusion_matrix(ground_truth, predicted_majority)

fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_full,
                              display_labels=display_labels)
disp.plot()
disp.figure_.savefig(f"/home/michal/thesis/dl_behaviour_model/plots/dlc/DLC8_{name_suffix}.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_majority,
                              display_labels=display_labels)
disp.plot()
disp.figure_.savefig(f"/home/michal/thesis/dl_behaviour_model/plots/dlc/DLC1_{name_suffix}.pdf", bbox_inches="tight")