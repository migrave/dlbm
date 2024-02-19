# data_preprocessing
This repository provides scripts for extracting sequences of eight frames from the annotated videos with:
* participants playing sequence learning game
* participants acting engaged/disengaged

There are four scripts that can be used for extracting frames, namely:
* for extracting sequences of frames from videos with people playing sequence learning game:
  * `frames_splitter.py` for normal frames
  * `zoomed_frames_splitter.py` for zoomed-in frames
* for extracting sequences of frames from videos with people acting engaged/disengaged:
  * `engagement_frames_splitter.py` for normal frames
  * `engagement_zoomed_frames_splitter.py` for zoomed-in frames.

### Requirements:
```
bagpy == 0.5
pandas == 1.5.3
numpy == 1.24.1
tqdm == 4.64.1
torch == 1.13.1
torchvision == 0.14.1
```