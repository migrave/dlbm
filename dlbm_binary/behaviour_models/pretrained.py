"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_binary.
      It is used for training and evaluation of the pretrained behaviour models.

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
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as resnet_weights
from torchvision.models.video import r2plus1d_18 as resnet3d
from torchvision.models.video import R2Plus1D_18_Weights as resnet3d_weights
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
        the pretrained ResNet networks.
    """
    def __init__(self, cfg, validation=False, input_state_size=1, rgb=False, name="resnet"):
        """
        :param cfg: configuration of the network
        :param validation: boolean information whether the model should be used in an evaluation mode (no gradients)
        :param input_state_size: number of frames as input
        :param rgb: boolean information whether the input image is RGB (true) or Grayscale (False)
        :param name: name of the model
        """
        torch.cuda.empty_cache()
        self.device = cfg.device
        self.state_dim = cfg.proc_frame_size
        self.state_size_data = cfg.state_size_data
        self.minibatch_size = cfg.minibatch_size
        self.learning_rate = cfg.learning_rate
        self.validation = validation
        self.epochs_num = cfg.epochs
        self.epoch_end = self.epochs_num
        self.epoch_start = 0
        self.input_state_size = input_state_size
        self.rgb = rgb
        self.name = name
        self.classes_num = cfg.noutputs

        checkpoint_path = f'/home/michal/thesis/dl_behaviour_model_binary/results/{self.name}/checkpoints'

        # model
        if self.input_state_size == 1:
            weights = resnet_weights.IMAGENET1K_V1
            self.model = resnet(weights=weights)
            self.preprocess = weights.transforms()

        elif self.input_state_size == 8:
            weights = resnet3d_weights.DEFAULT
            self.model = resnet3d(weights=weights)
            self.preprocess = weights.transforms()

        else:
            raise ValueError("Unaccountable state size")

        if not validation:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, cfg.noutputs)

        print(num_ftrs)
        self.model.to(self.device)

        self.trainloader, self.validloader, self.testloader = \
            self.load_dataset()

        print(f"Dataset training size: {len(self.trainloader.dataset)} \n"
              f"Dataset validation size: {len(self.validloader.dataset)} \n"
              f"Dataset test size: {len(self.testloader.dataset)}")

        checkpoint_name = 'model_epoch'
        pattern = re.compile(checkpoint_name + '[0-9]+.pt')

        # metrics
        self.accuracy = torchmetrics.classification.Accuracy(task='binary')
        self.precision = torchmetrics.classification.Precision(task='binary')
        self.recall = torchmetrics.classification.Recall(task='binary')
        self.f1_score = torchmetrics.classification.F1Score(task='binary')
        self.conf_mat = torchmetrics.classification.ConfusionMatrix(task='binary')

        params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
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
            params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters {params_count}")

        except Exception as err:
            print(repr(err))

    def load_dataset(self, path="/home/michal/thesis/interaction_dataset"):
        """
        Creating the dataset loaders
        :param path: path to the interaction dataset
        :return: loaders for the training, validation and test datasets
        """
        params = {'n_frames_used': self.input_state_size, 'rgb': True, 'standardization': False}
        train = InteractionDataset(path=os.path.join(path, 'train'), **params)
        valid = InteractionDataset(path=os.path.join(path, 'valid'), **params)
        test = InteractionDataset(path=os.path.join(path, 'test'), **params)

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
        plt.savefig(f"plots/{self.name}/training_loss_state_{self.input_state_size}.png")

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
                images = self.preprocess(images)

                images, labels = images.to(self.device), labels.to(self.device)
                sgray_action_values = self.model(images)

                self.optimizer.zero_grad()

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

                    images = self.preprocess(images)
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


            checkpoints_dir = 'checkpoints'

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
                image = self.preprocess(image)

                image, labels = image.to(self.device), label.to(self.device)
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
        disp.figure_.savefig(f"plots/{self.name}/confusion_matrix_state_{self.input_state_size}.png")

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
        tensor = self.preprocess(tensor)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            action_values = self.model(tensor)

        return torch.argmax(action_values.cpu().detach().clone(), dim=1)

    def majority_vote(self, tensor):
        """
        Majority vote
        :param tensor: input tensor (series of images 198x198)
        :return: action to be performed
        """
        if self.input_state_size != 1:
            raise 'Function unusable with input state size != 1'
        if not self.rgb:
            images = torch.unsqueeze(tensor[0], dim=1)
        else:
            images = tensor.squeeze()

        predicitions = self.predict(images)
        return np.bincount(predicitions).argmax()

