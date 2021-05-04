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


import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

vidcap = cv2.VideoCapture('videos/truck.mp4')

frame_counter = 0
motion_frames = []
# initialize the first frame in the video stream
first_frame = None

# loop over the frames of the video
while vidcap.isOpened():
    # grab the current frame and initialize the occupied/unoccupied text
    frame = vidcap.read()[1]
    # frame = frame if args.get("video", None) is None else frame[1]
    motion = False

    # if the frame could not be grabbed, then we have reached the end of the video
    if frame is None:
        break
    # resize the frame, convert it to grayscale, and blur it
    (h, w) = frame.shape[:2]
    width = 500
    r = width / float(w)
    dim = (width, int(h * r))
    cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if first_frame is None:
        first_frame = gray
        continue

    # compute the absolute difference between the current frame and first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion = True

    if motion:
        motion_frames.append(frame_counter)

    # Increment frame counter
    frame_counter += 1

    cv2.namedWindow("Security Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Security Feed", 1280, 720)
    cv2.imshow("Security Feed", frame)
    cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Thresh", 1280, 720)
    cv2.imshow("Thresh", thresh)
    cv2.namedWindow("Frame Delta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame Delta", 1280, 720)
    cv2.imshow("Frame Delta", frame_delta)

    cv2.waitKey(27)

# close any open windows
cv2.destroyAllWindows()
vidcap.release()
print(motion_frames)