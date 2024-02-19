#!/usr/bin/env python3

'''
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of data_preprocessing.
    It is used for extracting frames from the raw video data collected during the study
    where participants acted engaged/disengaged.

    data_preprocessing is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    data_preprocessing is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with data_preprocessing. If not, see <http://www.gnu.org/licenses/>.
'''

from bagpy import bagreader
import pandas as pd
import os
import os.path
import cv2
import numpy as np
import random
from tqdm import tqdm

# Directory with rosbag files containing timestamps indicating user's state transitions
BASE_PATH = "/home/michal/thesis/rosbags"
# Directory with the video files
VIDEO_BASE_PATH = "/media/michal/migrave/202203_hbrs_data_collection_experiment/qt"
# Directory with video files and frames' timestamps
TIMESTAMP_PATH = "engagement_estimation/video"
# Output directory for the extracted frame series
OUTPUT_PATH = "/home/michal/thesis/extra_frames"


def extract_frames(idx, frame_numbers, video_file):
    """
    Extract frmaes from a video file
    :param idx: User id
    :param frame_numbers: frame numbers indicating user's state transition
    :param video_file: name of the video file
    :return: frames
    """
    label_dict = {0: 'diff1', 1: 'diff1', 2: 'diff2', 3: 'diff2', 4: 'diff3', 5: 'diff3'}
    vidcap = cv2.VideoCapture(os.path.join(VIDEO_BASE_PATH, idx, TIMESTAMP_PATH, video_file))
    # engagement
    engaged_start = frame_numbers[0] + 1
    engaged_stop = frame_numbers[1] - 1
    # disengagement
    disengaged_start = frame_numbers[4] + 1
    disengaged_stop = frame_numbers[5] - 1

    for category, start, stop in [['diff', engaged_start, engaged_stop],
                                  ['feedback', disengaged_start, disengaged_stop]]:

        frames = get_frames(vidcap, start, stop)

        frame_series = np.array_split(frames, 6)

        for s_id, series in enumerate(frame_series):
            series_frames = filter_frames(series)

            for f_id, frame in enumerate(series_frames):
                if category == 'feedback':
                    path = os.path.join(OUTPUT_PATH, category, f"{idx}_extra_{s_id}_{f_id}.jpg")
                else:
                    path = os.path.join(OUTPUT_PATH, label_dict[s_id], f"{idx}_extra_{s_id}_{f_id}.jpg")

                squeezed_frame = frame.squeeze()
                cv2.imwrite(path, squeezed_frame)  # save frame as JPG file


def get_similarity(frame1, frame2):
    """
    Calculate similarity between two frames
    :param frame1:
    :param frame2:
    :return: similarity
    """
    diff = frame1 / 255 - frame2 / 255
    diff = np.around(diff, 2)
    nonzero_elem = np.count_nonzero(diff)
    total = diff.shape[0] * diff.shape[1]
    return 1. - nonzero_elem / total


def filter_frames(frames, size=8, threshold=0.8):
    """
    Filter frames so that only the least similar are left
    :param frames: frames to filter
    :param size: number of frames to be left
    :param threshold: similarity threshold (if similarity is below it, frames are not considered similar)
    :return: filtered frames
    """
    similarities = []

    for idx in range(0, frames.shape[0] - 1):
        similarity = get_similarity(frames[idx], frames[idx + 1])
        similarities.append(similarity)

    similarities = np.array(similarities)
    dissimilar_ids = np.argwhere(similarities < threshold)
    dissimilarities = (similarities[dissimilar_ids]).flatten()

    if not list(dissimilar_ids):
        random_frame_id = random.randint(0, frames.shape[0] - 1)
        random_frame = frames[random_frame_id]
        frames = np.array([random_frame] * size)
    else:
        if len(dissimilar_ids) < size:
            frames = frames[dissimilar_ids]
            last_frame = frames[-1]
            frames = list(frames)
            [frames.append(last_frame) for id in range(size - len(dissimilar_ids))]

        elif len(dissimilar_ids) == size:
            frames = frames[dissimilar_ids]

        else:
            indexes = np.argpartition(dissimilarities, size)[:size]
            indexes = np.sort(indexes)
            frames = frames[dissimilar_ids[indexes]]

    return frames


def get_frames(vidcap, start, end):
    """
    Extract frames from a video
    :param vidcap: object used to open a video file
    :param start_time: first frame occurrence time
    :param end_time: last frame occurrence time
    :return: extracted frames
    """
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    if not (start >= 0 & start <= total_frames & end >= 0 & end <= total_frames):
        return

    for idx in range(start, end + 1):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vidcap.read()
        frame = cv2.resize(frame, (198, 198))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    return np.array(frames)


def get_frame_nr(point, timestamps):
    """
    Finiding the closest timestamp to the given point in time
    :param point: point in time
    :param timestamps: list of timestamps
    :return: timestamp
    """
    for id, timestamp in enumerate(timestamps):
        if timestamp > point:
            return id - 1
    return len(timestamps) - 1


ids = os.listdir(BASE_PATH)

for idx in tqdm(ids):
    rosbags = [file for file in os.listdir(os.path.join(BASE_PATH, idx)) if file.split('.')[-1] == 'bag']
    rosbag = rosbags[0]
    b = bagreader(os.path.join(BASE_PATH, idx, rosbag))
    msg = b.message_by_topic('/migrave/engagement/event')
    df_msg = pd.read_csv(msg)

    points = df_msg["stamp.secs"].to_numpy()

    path = os.path.join(VIDEO_BASE_PATH, idx, TIMESTAMP_PATH)

    timestamps = [file for file in os.listdir(path) if "color" in file and ".txt" in file][0]
    video_name = [file for file in os.listdir(path) if "color" in file and ".mp4" in file][0]
    timestamps = np.loadtxt(os.path.join(path, timestamps))
    timestamps = [int(str(number)[:10]) for number in timestamps]

    frame_numbers = []

    for point in points:
        frame_nr = get_frame_nr(point, timestamps)
        frame_numbers.append(frame_nr)

    df_msg["frame"] = frame_numbers
    extract_frames(idx, frame_numbers, video_name)
