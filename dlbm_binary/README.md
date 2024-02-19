# dlbm_binary
This repository contains software for evaluation of four vision-only behaviour models on the binary MigrAVE dataset (feedback/no feedback).
The evaluated models are:
* DLC1 - Deep Learning Classifier with an input of one frame
* DLC8 - Deep Learning Classifier with an input of eight frames
* ResNet50 - fine-tuned image classification ResNet50 network with an input of one frame
* VideoResNet18 - fine-tuned video classification ResNet18 network with an input of eight frames

The implementation of this repository is based on [this source](https://github.com/JPedroRBelo/pyMDQN).

### Dataset

Training and evaluation of the aforementioned models is performed on the binary MigrAVE dataset collected, part of which was collected in [1]. 
This dataset should be created in the `interaction_dataset` directory with the following internal structure:
```
├── train
│   ├── diff
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
├── valid
│   ├── diff
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
├── test
│   ├── diff
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
```

To create `interaction_dataset` use `dataset_creator.py` script which needs `frames` directory with all the frame series (with already augmented background) with the following structure:
```
├── diff
│   ├── *.jpg
├── feedback
│   ├── *.jpg
```

```
[1] M. Stolarz, A. Mitrevski, M. Wasil, and P. G. Plöger, 
“Personalised Robot Behaviour Modelling for Robot-Assisted Therapy in the Context of Autism Spectrum Disorder,” 
in RO-MAN Workshop on Behavior Adaptation and Learning for Assistive Robotics (BAILAR), 2022.
```

### Requirements:
```
pandas == 1.5.3
numpy == 1.24.1
tqdm == 4.64.1
torch == 1.13.1
torchvision == 0.14.1
torchmetrics == 0.11.3
scikit-learn == 1.2.2
matplotlib == 3.7.0
Pillow == 9.4.0
opencv-python == 4.7.0.68
grad-cam == 1.4.6 
```

### Usage
* Dataset creation
  * To augment frames with new backgrounds use scripts in the `new_background_utils` directory, namely: `dataset_background_augment.py`, `background_augment_test.py`, `white_background_augment_test.py`.
  * To augment, balance and split the dataset into the training, validation and test sets, run the following command:
    ```
      python3 dataset_creator.py
    ```
  * To guarantee that the validation and test sets will have equal number of samples, run the following command:
    ```
      python3 dataset_equalizer.py
    ```
* Models deployment
  * For each model there are specified parameters in the configuration files in the `config` directory. 
  * There are different scripts to run the training procedure for different models. They are as follows:
    * DLC1/DLC8
      ```
      python3 train_dlc.py
      ```
    * ResNet50
      ```
      python3 train_resnet.py
      ```
    * VideoResNet18
      ```
      python3 train_resnet3d.py
      ```
    
    It should be mentioned that the trained models will be saved in the `results` directory, and the training results (plots, confusion matrices) will be saved in the `plots` directory.
  
  * There are also different scripts to evaluate already trained models. They are as follows:
    * DLC1 and DLC8
      ```
      python3 test_dlc.py
      ```
    * ResNet50 and VideoResnet18
      ```
      python3 test_resnet.py
      ```
    * To test all four models on the test sets with five new backgrounds or without a background, use scripts in the `new_background_utils` directory, namely: `test_new_backgrounds.py`, `test_resnet_new_backgrounds.py`.
    
    It should be mentioned that the training results (plots, confusion matrices) will be saved in the `plots` directory.
  
  * To generate Grad-CAM++ heatmaps following jupyter notebooks should be used:
    * `gradcampp_dlc1.ipynb` for DLC1
    * `gradcampp_dlc8.ipynb` for DLC8
    * `gradcampp_resnet.ipynb` for ResNet50
    * `gradcampp_resnet3d.ipynb` for VideoResNet18

* Extra scripts in the `utils` directory
  * To obtain mean and standard deviation values for image standardisation run:
    ```
    python3 trainset_stats.py
    ```
  * To plot validation and training loss for each model run:
    ```
    python3 training_stats.py
    ```
  * To plot statistics for the used dataset, use `dataset_check.ipynb`.
