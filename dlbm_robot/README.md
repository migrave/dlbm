# dlbm_robot
This repository contains a ROS Noetic package for deploying a deep learning-based behaviour model on the robot for a sequence learning game. 
This game works only with a respective application developed in QTrobot Studio.

To be launched an additional package is required, namely [migrave_ros_msgs](https://github.com/migrave/migrave_ros_msgs).

### Requirements:
```
numpy == 1.24.1
torch == 2.1.2
torchvision == 0.16.2
cv_bridge == 1.15.0
collections == 1.2.2
opencv-python == 4.9.0.80
```

### Usage
To start sequence learning game with a deep learning-based behaviour model run:
```
roslaunch dlbm_robot demo_game_manager.launch
```
