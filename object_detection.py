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


from cvlib import detect_common_objects
from cv2 import imread
from cvlib.object_detection import draw_bbox


class ObjectDetection:
    def __init__(self, frames, frames_dir):
        self.frames_dir = frames_dir
        self.frames = frames
        self.objects = []

    def detect(self):
        for frame in self.frames:
            image = imread(f'{self.frames_dir}{frame}.jpg')
            if image is None:
                pass
            else:
                # apply object detection
                bbox, label, conf = detect_common_objects(image)

                # draw bounding box over detected objects
                out = draw_bbox(image, bbox, label, conf)
                if len(label) > 0:
                    self.objects.append((frame, out, label, conf))
        return self.objects
