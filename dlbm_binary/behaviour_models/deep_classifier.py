"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_binary.
    It is used for training and evaluation of the behaviour models trained from scratch on the binary MigrAVE dataset.

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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from models.dlc.network import DLC
from config import config_dlc as dcfg
from tqdm import tqdm
from utils.data_loader import InteractionDataset
import torchmetrics
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

torch.manual_seed(42)


class DeepClassifier:
    """
        Class used for training and evaluation of the behaviour models based on
        the deep learning classifier trained from scratch.
    """
    def __init__(self, cfg=dcfg, input_state_size=1):
        """
        :param cfg: configuration of the network
        :param input_state_size: number of frames as input
        """
        # cpu or cuda
        torch.cuda.empty_cache()
        self.device = cfg.device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = cfg.proc_frame_size  # State dimensionality 84x84.
        self.state_size_data = cfg.state_size_data
        self.minibatch_size = cfg.minibatch_size
        self.learning_rate = cfg.learning_rate
        self.epochs_num = cfg.epochs
        self.epoch_end = self.epochs_num
        self.epoch_start = 0
        self.input_state_size = input_state_size
        self.classes_num = cfg.noutputs

        if self.input_state_size == 1:
            nstates = cfg.nstates
            nfeats = cfg.nfeats
            checkpoint_path = '/home/michal/thesis/dl_behaviour_model_binary/results/dlc/checkpoints'

        elif self.input_state_size == 8:
            nstates = cfg.nstates_full
            nfeats = cfg.nfeats_full
            checkpoint_path = '/home/michal/thesis/dl_behaviour_model_binary/results/dlc/checkpoints_full_state'
        else:
            raise ValueError("Unaccountable state size")

        self.trainloader, self.validloader, self.testloader = \
            self.load_dataset()

        print(f"Dataset training size: {len(self.trainloader.dataset)} \n"
              f"Dataset validation size: {len(self.validloader.dataset)} \n"
              f"Dataset test size: {len(self.testloader.dataset)}")

        self.model = DLC(noutputs=cfg.noutputs, nfeats=nfeats,
                         nstates=nstates, kernels=cfg.kernels,
                         strides=cfg.strides, poolsize=cfg.poolsize).to(self.device)

        checkpoint_name = 'model_epoch'
        pattern = re.compile(checkpoint_name + '[0-9]+.pt')

        # metrics
        self.accuracy = torchmetrics.classification.Accuracy(task='binary')
        self.precision = torchmetrics.classification.Precision(task='binary')
        self.recall = torchmetrics.classification.Recall(task='binary')
        self.f1_score = torchmetrics.classification.F1Score(task='binary')
        self.conf_mat = torchmetrics.classification.ConfusionMatrix(task='binary')

        params_count = self.model.get_params_count()
        print(f"Number of trainable parameters {params_count}")

        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.loss_fuction = nn.CrossEntropyLoss()

        try:
            if os.path.exists(checkpoint_path):
                entries = sorted(os.listdir(checkpoint_path), reverse=True)
                if entries:
                    numbers = [int(re.findall('\d+', entry)[0]) for entry in entries if pattern.match(entry)]
                    epoch = max(numbers)

                    self.epoch_end = epoch + self.epochs_num + 1
                    self.epoch_start = epoch + 1

                    checkpoint = torch.load(os.path.join(checkpoint_path, f"{checkpoint_name}{epoch}.pt"))
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        except Exception as err:
            print(repr(err))

    def load_dataset(self, path="/home/michal/thesis/interaction_dataset"):
        """
        Creating the dataset loaders
        :param path: path to the interaction dataset
        :return: loaders for the training, validation and test datasets
        """
        train = InteractionDataset(path=os.path.join(path, 'train'), n_frames_used=self.input_state_size)
        valid = InteractionDataset(path=os.path.join(path, 'valid'), n_frames_used=self.input_state_size)
        test = InteractionDataset(path=os.path.join(path, 'test'), n_frames_used=self.input_state_size)

        trainloader = torch.utils.data.DataLoader(train,
                                                  batch_size=self.minibatch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        validloader = torch.utils.data.DataLoader(valid,
                                                  batch_size=self.minibatch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        testloader = torch.utils.data.DataLoader(test,
                                                 batch_size=1,
                                                 shuffle=True)

        return trainloader, validloader, testloader

    def plot_loss(self, train_loss_all, valid_loss_all):
        """
        Plotting losses
        :param train_loss_all: list with the training loss for each epoch
        :param valid_loss_all: list with the validation loss for each epoch
        :return: None
        """
        fig, ax = plt.subplots(1, 1)
        x = np.arange(1, len(train_loss_all) + 1)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(x, train_loss_all, label='training')
        plt.plot(x, valid_loss_all, label='validation')
        plt.xticks(x)
        plt.xlim((0.95, self.epochs_num + 0.05))
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/dlc/training_loss_state_{self.input_state_size}.png")

    def train_model(self, path):
        """
        Training the model
        :param path: path to save the model checkpoints
        :return: None
        """
        train_loss_all = []
        valid_loss_all = []

        for epoch in tqdm(range(self.epoch_start, self.epoch_end)):
            running_train_loss = 0.0
            running_valid_loss = 0.0

            self.model.train()

            for images, labels in self.trainloader:
                self.optimizer.zero_grad()

                images, labels = images.to(self.device), labels.to(self.device)

                sgray_action_values = self.model(images)
                train_loss = self.loss_fuction(sgray_action_values, labels)
                train_loss.backward()
                self.optimizer.step()

                running_train_loss += train_loss.item()

            self.model.eval()

            predicted = []
            ground_truth = []

            with torch.no_grad():
                for images, labels in self.validloader:
                    ground_truth.append(labels.clone())
                    images, labels = images.to(self.device), labels.to(self.device)

                    action_values = self.model(images)
                    valid_loss = self.loss_fuction(action_values, labels)
                    running_valid_loss += valid_loss.item()

                    predicted.append(torch.argmax(action_values.cpu().detach().clone(), dim=1))

            mean_train_loss = running_train_loss / len(self.trainloader)
            mean_valid_loss = running_valid_loss / len(self.validloader)

            prediction_labels = torch.cat(predicted)
            gt_labels = torch.cat(ground_truth)

            accuracy_score = self.accuracy(prediction_labels, gt_labels)
            recall_score = self.recall(prediction_labels, gt_labels)
            precision_score = self.precision(prediction_labels, gt_labels)
            f1score = self.f1_score(prediction_labels, gt_labels)

            train_loss_all.append(mean_train_loss)
            valid_loss_all.append(mean_valid_loss)

            print(f'[Epoch {epoch}] training loss: {mean_train_loss:.3f}, '
                  f'validation loss: {mean_valid_loss:.3f}, '
                  f'accuracy: {accuracy_score:.3f}, '
                  f'precision: {precision_score:.3f}, '
                  f'recall: {recall_score:.3f}, '
                  f'f1 score: {f1score:.3f}')

            if self.input_state_size == 1:
                checkpoints_dir = 'checkpoints'
            elif self.input_state_size == 8:
                checkpoints_dir = 'checkpoints_full_state'

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss_last': train_loss.item(),
                'train_loss_avg': mean_train_loss,
                'val_loss_avg': mean_valid_loss,
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1score,
            }, os.path.join(path, checkpoints_dir, f'model_epoch{epoch}.pt'))

        self.plot_loss(train_loss_all, valid_loss_all)

    def test_model(self):
        """
        Testing the model
        :param None
        :return: None
        """
        self.model.eval()

        predicted = []
        ground_truth = []

        with torch.no_grad():
            for image, label in self.testloader:
                ground_truth.append(label.clone())
                image = image.to(self.device)

                action_values = self.model(image)
                predicted.append(torch.argmax(action_values.cpu().detach().clone(), dim=1))

        prediction_labels = torch.cat(predicted)
        gt_labels = torch.cat(ground_truth)

        accuracy_score = self.accuracy(prediction_labels, gt_labels)
        recall_score = self.recall(prediction_labels, gt_labels)
        precision_score = self.precision(prediction_labels, gt_labels)
        f1score = self.f1_score(prediction_labels, gt_labels)
        # conf_matrix = self.conf_mat(prediction_labels, gt_labels)
        conf_matrix = confusion_matrix(gt_labels, prediction_labels)

        print(f'Accuracy: {accuracy_score:.3f},\n'
              f'Recall: {recall_score:.3f},\n'
              f'Precision: {precision_score:.3f},\n'
              f'F1 score: {f1score:.3f},\n'
              f'Conf matrix: {conf_matrix},\n')

        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            #display_labels=['wait', 'wave', 'hand_shake'])
            display_labels=['diff', 'feedback'])
        disp.plot()
        disp.figure_.savefig(f"plots/dlc/confusion_matrix_state_{self.input_state_size}.png")

    def train(self, path):
        """
        Training and evaluation of the model
        :param None
        :return: None
        """
        self.train_model(path)
        self.test_model()

    def predict(self, tensor):
        """
        Predicting actions with the model
        :param tensor: input tensor (image or series of images 198x198)
        :return: action to be performed
        """
        self.model.eval()

        with torch.no_grad():
            tensor_device = tensor.to(self.device)
            action_values = self.model(tensor_device)

        return torch.argmax(action_values.cpu().detach().clone(), dim=1)

    def majority_vote(self, tensor):
        """
        Majority vote
        :param tensor: input tensor (series of images 198x198)
        :return: action to be performed
        """
        if self.input_state_size != 1:
            raise 'Function unusable with input state size == 1'
        images = torch.unsqueeze(tensor[0], dim=1)
        predicitions = self.predict(images)
        return np.bincount(predicitions).argmax()

