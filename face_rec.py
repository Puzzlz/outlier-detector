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


import face_recognition
import os


class FaceRec:
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir

    def scan(self):
        # TODO
        # What happens if there are 2 faces in a frame, 1 known and the other not?
        # Need a way of detecting how many faces are in a frame.
        # image = face_recognition.load_image_file("frames/walkway/152.jpg")
        # face_locations = face_recognition.face_locations(image)
        # num_faces = len(face_locations)
        results = []
        frames = os.listdir(self.frames_dir)
        # Convert to ints so it will process video in order of frames
        # Will be useful to compare to previous and following frames for known faces
        frames = [int(x.split('.')[0]) for x in frames]
        frames.sort()
        known_encodings = []
        faces = os.listdir('known_faces')
        for friendly_face in faces:
            enc = face_recognition.load_image_file(f'known_faces/{friendly_face}')
            try:
                known_encodings.append(face_recognition.face_encodings(enc)[0])
            except IndexError:
                continue

        # Check frames for familiar faces
        for frame in frames:
            for face in known_encodings:
                # Only check every 8 frames to get quicker results
                if frame % 8 == 0:
                    try:
                        unknown_image = face_recognition.load_image_file(f'{self.frames_dir}{str(frame)+".jpg"}')
                        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                    # If a frame doesn't contain a recognizable face it errors out, this will allow it to continue
                    except IndexError:
                        continue
                    result = face_recognition.compare_faces([face], unknown_encoding)
                    results.append((result[0], frame))
        # TODO Handle the frames being flagged
        return results
