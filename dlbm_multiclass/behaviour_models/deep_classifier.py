"""
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of dlbm_multiclass.
    It is used for training and evaluation of the behaviour models trained from scratch on the multiclass MigrAVE dataset.

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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from models.dlc.network import DLC
from config import config_dlc as dcfg
from tqdm import tqdm
from utils.data_loader import InteractionDataset
from utils.dataset_utils import balance_dataset
import torchmetrics
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy

torch.manual_seed(42)


class DeepClassifier:
    """
        Class used for training and evaluation of the behaviour models based on
        the deep learning classifier trained from scratch.
    """
    def __init__(self, cfg=dcfg, input_state_size=1, nfold=False):
        """
        :param cfg: configuration of the network
        :param input_state_size: number of frames as input
        :param nfold: boolean variable indicating whether n-fold cross-validation should be performed
        """
        torch.cuda.empty_cache()
        self.device = cfg.device
        self.state_dim = cfg.proc_frame_size
        self.state_size_data = cfg.state_size_data
        self.minibatch_size = cfg.minibatch_size
        self.learning_rate = cfg.learning_rate
        self.epochs_num = cfg.epochs
        self.epoch_end = self.epochs_num
        self.epoch_start = 0
        self.input_state_size = input_state_size
        self.classes_num = cfg.noutputs
        self.conf_matrix_labels = cfg.labels
        self.nfold = nfold
        self.model_noutputs = cfg.noutputs
        self.model_kernels = cfg.kernels
        self.model_strides = cfg.strides
        self.model_poolsize = cfg.poolsize

        # model
        if self.input_state_size == 1:
            self.model_nstates = cfg.nstates
            self.model_nfeats = cfg.nfeats
            checkpoint_path = '/home/michal/thesis/dl_behaviour_model_multiclass/results/dlc/checkpoints'

        elif self.input_state_size == 8:
            self.model_nstates = cfg.nstates_full
            self.model_nfeats = cfg.nfeats_full
            checkpoint_path = '/home/michal/thesis/dl_behaviour_model_multiclass/results/dlc/checkpoints_full_state'
        else:
            raise ValueError("Unaccountable state size")

        self.model, self.optimizer = self.load_new_model(learning_rate = self.learning_rate)

        # metrics
        self.accuracy = torchmetrics.classification.Accuracy(task='multiclass', average='macro',
                                                             num_classes=self.classes_num)
        self.precision = torchmetrics.classification.Precision(task="multiclass", average='macro',
                                                               num_classes=self.classes_num)
        self.recall = torchmetrics.classification.Recall(task="multiclass", average='macro',
                                                         num_classes=self.classes_num)
        self.f1_score = torchmetrics.classification.F1Score(task="multiclass", average='macro',
                                                            num_classes=self.classes_num)
        self.conf_mat = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=self.classes_num)

        params_count = self.model.get_params_count()
        print(f"Number of trainable parameters {params_count}")
        self.loss_fuction = nn.CrossEntropyLoss()

        if self.nfold:
            self.usrs_datasets = self.load_nfold_dataset()
            return

        self.trainloader, self.validloader, self.testloader = \
            self.load_dataset()  # path='/media/michal/migrave/3labels/interaction_dataset'
        print(f"Dataset training size: {len(self.trainloader.dataset)} \n"
              f"Dataset validation size: {len(self.validloader.dataset)} \n"
              f"Dataset test size: {len(self.testloader.dataset)}")

        checkpoint_name = 'model_epoch'
        pattern = re.compile(checkpoint_name + '[0-9]+.pt')

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

    def load_new_model(self, learning_rate):
        """
        Loading a behaviour model
        :param learning_rate: learning rate
        :return: model, optimizer
        """
        model = DLC(noutputs=self.model_noutputs, nfeats=self.model_nfeats,
                    nstates=self.model_nstates, kernels=self.model_kernels,
                    strides=self.model_strides, poolsize=self.model_poolsize,
                    enable_activity_signals=True).to(self.device)
        optimizer = optim.Adam(model.parameters(), learning_rate)
        return model, optimizer

    def load_nfold_dataset(self, path="/home/michal/thesis/interaction_nfold_dataset"):
        """
        Loading dataset splits for each user for n-fold cross-validation
        :param path: path to data
        :return: dataset splits
        """
        dirs = os.listdir(path)
        usrs_datasets = {}

        for usr in dirs:
            usrs_datasets[usr] = InteractionDataset(path=os.path.join(path, usr),
                                                    n_frames_used=self.input_state_size)

        return usrs_datasets

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

    def merge_datasets(self, datasets):
        dataset = copy.deepcopy(datasets[0])
        img_labels = []
        img_paths = []

        for usr_dataset in datasets:
            img_labels = img_labels + list(usr_dataset.img_labels)
            img_paths = img_paths + usr_dataset.img_paths

        dataset.img_labels = np.array(img_labels)
        dataset.img_paths = img_paths
        return dataset

    def check_left_users(self, path):
        checkpoints_path = os.path.join(path, 'nfold')
        users = set(os.listdir(checkpoints_path))
        all_users = set(self.usrs_datasets.keys())
        return all_users - users

    def train_nfold_model(self, path):
        """
        Perform n-fold cross-validation for every user
        :param path: path to save the model checkpoints
        :return: None
        """
        users = self.check_left_users(path)

        for usr in users:
            self.model, self.optimizer = self.load_new_model(learning_rate=self.learning_rate)

            print('USER: ', usr)
            os.mkdir(os.path.join(path, 'nfold', usr))

            training_set = [dataset for key, dataset in self.usrs_datasets.items() if key != usr]

            train = self.merge_datasets(training_set)
            train = balance_dataset(train)
            test = self.usrs_datasets[usr]

            trainloader = torch.utils.data.DataLoader(train,
                                                      batch_size=self.minibatch_size,
                                                      shuffle=True,
                                                      num_workers=2)
            testloader = torch.utils.data.DataLoader(test,
                                                     batch_size=1,
                                                     shuffle=True)

            print(f"Dataset training size: {len(trainloader.dataset)} \n"
                  #      f"Dataset validation size: {len(validloader.dataset)} \n"
                  f"Dataset test size: {len(testloader.dataset)}")

            self.train_model(path, 0, self.epoch_end, trainloader, testloader,
                             nfold_path=os.path.join('nfold', usr))

    def train_model(self, path, start_epoch, end_epoch, trainloader, validloader, nfold_path=None):
        """
        Training the model
        :param path: path to save the model checkpoints
        :param start_epoch: first training epoch
        :param end_epoch: last training epoch
        :param trainloader: loader of the training dataset
        :param validloader: loader of the validation dataset
        :param nfold_path: path to save the model checkpoints for every user (n-fold cross-validation)
        :return: None
        """
        train_loss_all = []
        valid_loss_all = []

        for epoch in tqdm(range(start_epoch, end_epoch)):
            running_train_loss = 0.0
            running_valid_loss = 0.0

            self.model.train()

            for state, labels in trainloader:
                images, activity = state
                self.optimizer.zero_grad()

                images, activity, labels = images.to(self.device), activity.to(self.device), labels.to(self.device)
                full_state = [images, activity]

                sgray_action_values = self.model(full_state)
                train_loss = self.loss_fuction(sgray_action_values, labels)
                train_loss.backward()
                self.optimizer.step()

                running_train_loss += train_loss.item()

            self.model.eval()

            predicted = []
            ground_truth = []

            with torch.no_grad():
                for state, labels in validloader:
                    images, activity = state
                    ground_truth.append(labels.clone())

                    images, activity, labels = images.to(self.device), activity.to(self.device), labels.to(self.device)

                    full_state = [images, activity]
                    action_values = self.model(full_state)
                    valid_loss = self.loss_fuction(action_values, labels)
                    running_valid_loss += valid_loss.item()

                    predicted.append(torch.argmax(action_values.cpu().detach().clone(), dim=1))

            mean_train_loss = running_train_loss / len(trainloader)
            mean_valid_loss = running_valid_loss / len(validloader)

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

            if not nfold_path:
                checkpoint_path = os.path.join(path, checkpoints_dir, f'model_epoch{epoch}.pt')
            else:
                checkpoint_path = os.path.join(path, nfold_path, f'model_epoch{epoch}.pt')

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
            }, checkpoint_path)

            # early stopping
            if nfold_path and len(valid_loss_all) > 2:
                if mean_valid_loss >= valid_loss_all[-3] and valid_loss_all[-2] >= valid_loss_all[-3]:
                    break

        self.plot_loss(train_loss_all, valid_loss_all)

    def test_model(self, testloader):
        """
        Testing the model
        :param testloader: loader of the test dataset
        :return: None
        """
        self.model.eval()

        predicted = []
        ground_truth = []

        with torch.no_grad():
            for state, label in testloader:
                image, activity = state
                ground_truth.append(label.clone())
                image, activity = image.to(self.device), activity.to(self.device)

                full_state = [image, activity]
                action_values = self.model(full_state)
                predicted.append(torch.argmax(action_values.cpu().detach().clone(), dim=1))

        prediction_labels = torch.cat(predicted)
        gt_labels = torch.cat(ground_truth)

        accuracy_score = self.accuracy(prediction_labels, gt_labels)
        recall_score = self.recall(prediction_labels, gt_labels)
        precision_score = self.precision(prediction_labels, gt_labels)
        f1score = self.f1_score(prediction_labels, gt_labels)
        conf_matrix = confusion_matrix(gt_labels, prediction_labels)

        print(f'Accuracy: {accuracy_score:.3f},\n'
              f'Recall: {recall_score:.3f},\n'
              f'Precision: {precision_score:.3f},\n'
              f'F1 score: {f1score:.3f},\n'
              f'Conf matrix: {conf_matrix},\n')

        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=self.conf_matrix_labels)
        disp.plot()
        disp.figure_.savefig(f"plots/dlc/confusion_matrix_state_{self.input_state_size}.png")

    def train(self, path):
        """
        Evaluation of the model (training and testing)
        :param path: path to save the model checkpoints
        :return: None
        """
        if not self.nfold:
            self.train_model(path, self.epoch_start, self.epoch_end, self.trainloader, self.validloader)
            self.test_model(self.testloader)
        else:
            self.train_nfold_model(path)

    def predict(self, tensor):
        """
        Predicting actions with the model
        :param tensor: input tensor (image or series of images 198x198)
        :return: action to be performed
        """
        self.model.eval()
        images, activity = tensor

        images, activity = images.to(self.device), activity.to(self.device)
        full_state = [images, activity]

        with torch.no_grad():
            action_values = self.model(full_state)

        return torch.argmax(action_values.cpu().detach().clone(), dim=1)

    def majority_vote(self, tensor):
        """
        Majority vote
        :param tensor: input tensor (series of images 198x198)
        :return: action to be performed
        """
        images, activity = tensor

        if self.input_state_size != 1:
            raise 'Function unusable with input state size != 1'

        images = torch.unsqueeze(images[0], dim=1)

        full_state = [images, activity.repeat(images.shape[0], 1)]
        predictions = self.predict(full_state)

        return np.bincount(predictions).argmax()
