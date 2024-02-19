# dl_behaviour_model_multiclass
This repository contains software for evaluation of four vision-only behaviour models on the multiclass MigrAVE dataset (diff2, diff3, feedback).
The evaluated models are:
* DLC1 - Deep Learning Classifier with an input of one frame and an activity vector
* DLC8 - Deep Learning Classifier with an input of eight frames and an activity vector

The implementation of this repository is based on [this source](https://github.com/JPedroRBelo/pyMDQN).

### Dataset
Training and evaluation of the aforementioned models is performed on the multiclass MigrAVE dataset collected, part of which was collected in [1]. 
This dataset should be created in the `interaction_dataset` directory with the following internal structure:
```
├── train
│   ├── diff2
│   │   ├── *.jpg
│   ├── diff3
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
├── valid
│   ├── diff2
│   │   ├── *.jpg
│   ├── diff3
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
├── test
│   ├── diff2
│   │   ├── *.jpg
│   ├── diff3
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
```
 
One can also perform n-fold cross-validation. In that case one should create appropriate dataset in the  `interaction_nfold_dataset` directory with the following internal structure:
```
├── user_id_1
│   ├── diff2
│   │   ├── *.jpg
│   ├── diff3
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
├── user_id_2
│   ├── diff2
│   │   ├── *.jpg
│   ├── diff3
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
├── user_id_3
│   ├── diff2
│   │   ├── *.jpg
│   ├── diff3
│   │   ├── *.jpg
│   ├── feedback
│   │   ├── *.jpg
:
```
To create `interaction_dataset` use `dataset_creator.py` script. To create `interaction_nfold_dataset` use `n_fold_dataset_creator.py` script. Both scripts need `frames` directory with all frame series (with already augmented background) with the following structure:
```
├── diff2
│   ├── *.jpg
├── diff3
│   ├── *.jpg
├── feedback
│   ├── *.jpg
```
Lastly, it should be metioned, that there is required user's sequence learning game performance located in the 'csv_files' directory, where for each user there is a csv file containing this information.

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
```

### Usage
* Dataset creation
  * To augment frames with new backgrounds use scripts in the `new_background_utils` directory, namely: `dataset_background_augment.py`, `background_augment_test.py`, `white_background_augment_test.py`.
  * To augment, balance and split the dataset into the training, validation and test sets, run the following command:
    ```
      python3 dataset_creator.py
    ```
    To guarantee that the validation and test sets will have equal number of samples, run the following command:
    ```
      python3 dataset_equalizer.py
    ```
  * To create dataset for n-fold cross-validation run:
    ```
      python3 nfold_dataset_creator.py
    ```
* Models deployment
  * For each model there are specified parameters in the configuration files in the `config` directory. 
  * There are different scripts to run different training procedures. They are as follows:
    * DLC1/DLC8
      ```
      python3 train_dlc.py
      ```
    * DLC1/DLC8 using n-fold cross-validation
      ```
      python3 train_nfold_dlc.py
      ```
    
    It should be mentioned that the trained models will be saved in the `results` directory, and the training results (plots, confusion matrices) will be saved in the `plots` directory.
  
  * There are also different scripts for different evaluation of the models. They are as follows:
    * DLC1 and DLC8
      ```
      python3 test_dlc.py
      ```
    * To test all four models on the test sets with five new backgrounds or without a background, use a script in the `new_background_utils` directory, namely, `test_new_backgrounds.py`.
    
    It should be mentioned that the training results (plots, confusion matrices) will be saved in the `plots` directory.

* Extra scripts in the `utils` directory
  * To obtain mean and standard deviation values for image standardisation run:
    ```
    python3 trainset_stats.py
    ```
  * Plot F1 scores for every user:
    ```
    python3 training_stats.py
    ```
  * To plot statistics for the used dataset, use `dataset_check.ipynb`.
  * To plot F1 scores for every user (n-fold cross-validation) and print average performance metrics run: 
    ```
    python3 training_nfold_stats.py
    ```
