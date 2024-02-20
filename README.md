# migrave_dlbm
This repository contains software for development, training and evaluation of deep learning behaviour models (dlbm) for the sequence learning game.
This repository contains the following modules:
* `dlbm_data_preprocessing` - extracting frames from the annotated videos in order to create the datasets used in `dlbm_binary` and `multiclass_binary`
* `dlbm_binary` - software for training and evaluation of four vision-only classifiers on the binary MigrAVE dataset
* `dlbm_multiclass` - software for training and evaluation of two behaviour models on the multiclass MigrAVE dataset
* `dlbm_robot` - softare for deploying a behaviour model on the robot for a sequence learning game