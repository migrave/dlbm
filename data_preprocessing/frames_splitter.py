#!/usr/bin/env python3

'''
    Copyright 2023 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of data_preprocessing.
    It is used for extracting frames from the raw video data collected during the study
    where participants played sequence learning game.

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

import json
import os.path
import cv2
import numpy as np
import random
from tqdm import tqdm

# Output directory for the extracted frame series
OUTPUT_PATH = '/home/michal/thesis/frames'

# If MDOE = 1, first set of data is used, if MODE = 2, second set of data is used
MODE = 1 # 1 - 1 set of data, 2 - 2 second set of data

if MODE == 1:
    # File containing annotations of the videos
    JSON_PATH = 'boris_annotations/sequence_learning.boris'
else:
    # File containing annotations of the videos
    JSON_PATH = 'boris_annotations/sequence_learning_2.boris'


def get_similarity(frame1, frame2):
    """
    Calculate similarity between two frames
    :param frame1:
    :param frame2:
    :return: similarity
    """
    diff = frame1/255 - frame2/255
    diff = np.around(diff, 2)
    nonzero_elem = np.count_nonzero(diff)
    total = diff.shape[0]*diff.shape[1]
    return 1.-nonzero_elem/total


def filter_frames(frames, size=8, threshold=0.8):
    """
    Filter frames so that only the least similar are left
    :param frames: frames to filter
    :param size: number of frames to be left
    :threshold: similarity threshold (if similarity is below it, frames are not considered similar)
    :return: filtered frames
    """
    similarities = []

    for idx in range(0, frames.shape[0]-1):
        similarity = get_similarity(frames[idx], frames[idx+1])
        similarities.append(similarity)

    similarities = np.array(similarities)
    dissimilar_ids = np.argwhere(similarities < threshold)
    dissimilarities = (similarities[dissimilar_ids]).flatten()

    if not list(dissimilar_ids):
        random_frame_id = random.randint(0, frames.shape[0]-1)
        random_frame = frames[random_frame_id]
        frames = np.array([random_frame]*size)
    else:
        if len(dissimilar_ids) < size:
            frames = frames[dissimilar_ids]
            last_frame = frames[-1]
            frames = list(frames)
            [frames.append(last_frame) for id in range(size-len(dissimilar_ids))]

        elif len(dissimilar_ids) == size:
            frames = frames[dissimilar_ids]

        else:
            indexes = np.argpartition(dissimilarities, size)[:size]
            indexes = np.sort(indexes)
            frames = frames[dissimilar_ids[indexes]]

    return frames


def get_frame_number(vidcap, time):
    """
    Get number of a frame in the video
    :param vidcap: object used to open a video file
    :param time: frame occurrence time
    :return: frame number
    """
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time * 1000.)
    frame_position = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
    return frame_position


def get_frames(vidcap, start_time, end_time):
    """
    Extract frames from a video
    :param vidcap: object used to open a video file
    :param start_time: first frame occurrence time
    :param end_time: last frame occurrence time
    :return: extracted frames
    """
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = int(get_frame_number(vidcap, start_time))
    end = int(get_frame_number(vidcap, end_time))
    frames = []

    if not (start >= 0 & start <= total_frames & end >= 0 & end <= total_frames):
        return

    for idx in range(start, end+1):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vidcap.read()
        frame = cv2.resize(frame, (198, 198))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    return np.array(frames)


with open(JSON_PATH) as file:
    json_dict = json.load(file)
    for observation in tqdm(json_dict['observations']):
        file = json_dict['observations'][observation]['file']['1'][0].split('/')[-1]

        if MODE == 1:
            #file_path = f"/home/mstolarz/Encfs/migrave_dataset/to_annotate/prepared_data_migrave/{file}"
            file_path = f"/media/michal/misiek/to_annotate/prepared_data_migrave/{file}"
        else:
            code = file.split('_')[0]
            #file_path = f"/home/mstolarz/Encfs/migrave_dataset/to_annotate/prepared_data_migrave_2/{code}/{file}"
            file_path = f"/media/michal/misiek/to_annotate/prepared_data_migrave_2/{code}/{file}"

        vidcap = cv2.VideoCapture(file_path)

        events = json_dict['observations'][observation]['events']

        for idx in range(0, len(events), 2):
            name = events[idx][2]
            start = events[idx][0]
            end = events[idx+1][0]
            frames = get_frames(vidcap, start, end)
            frames = filter_frames(frames)

            for f_id, frame in enumerate(frames):
                path = os.path.join(OUTPUT_PATH, name, f"{observation}_{idx}_{f_id}.jpg")
                squeezed_frame = frame.squeeze()
                cv2.imwrite(path, squeezed_frame)  # save frame as JPG file