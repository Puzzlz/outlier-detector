# Outlier Detector, detect anomilies in a video
# Copyright (C) 2020  Tyson Steele

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


import cv2
import os
import math
import numpy as np


def diff_two_colours(colour1, colour2):
    diff = math.sqrt(
        abs((colour2[0] - colour1[0]) ** 2) + abs((colour2[1] - colour1[1]) ** 2) + abs((colour2[2] - colour1[2]) ** 2))
    return (diff / math.sqrt((255) ** 2 + (255) ** 2 + (255) ** 2)) * 100


def get_average_colour_frame(frames_dir, frame_file):
    # returns (R, G, B)
    my_img = cv2.imread(f'{frames_dir}{frame_file}')
    avg_color_per_row = np.average(my_img, axis=0)
    avg_colour = np.average(avg_color_per_row, axis=0)
    return avg_colour[2], avg_colour[1], avg_colour[0]


def get_oddly_coloured_frames(vid_avg_diff_frame_avg, cutoff):
    odd_frames = []

    for item in vid_avg_diff_frame_avg.items():
        if item[1] > cutoff:
            odd_frames.append(int(item[0].split('.')[0]))
    return odd_frames


# Source: https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
def colour_investigation(frames_dir, cutoff):
    video_average_colour = (0, 0, 0)
    vid_avg_diff_frame_avg = {}
    frame_counter = 0

    frames = os.listdir(frames_dir)

    # Get the videos average colour
    for frame in frames:
        frame_counter += 1
        avg_colour = get_average_colour_frame(frames_dir, frame)
        video_average_colour = (video_average_colour[0] + avg_colour[0], video_average_colour[1] + avg_colour[1],
                                video_average_colour[2] + avg_colour[2])
    video_average_colour = (video_average_colour[0] / frame_counter, video_average_colour[1] / frame_counter,
                            video_average_colour[2] / frame_counter)

    # Loop through the frames to find interesting ones based on colour
    for frame in frames:
        avg_colour = get_average_colour_frame(frames_dir, frame)
        vid_avg_diff_frame_avg[frame] = diff_two_colours(avg_colour, video_average_colour)
    return get_oddly_coloured_frames(vid_avg_diff_frame_avg, cutoff)


class ColourDetection:
    def __init__(self, frames_dir, cutoff):
        self.frames_dir = frames_dir
        self.cutoff = cutoff

    def get_frames(self):
        return colour_investigation(self.frames_dir, self.cutoff)
