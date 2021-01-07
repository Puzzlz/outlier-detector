# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
import cv2


class MotionDetection:
    def __init__(self, video, min_area=5000):
        self.vidcap = cv2.VideoCapture(video)
        # 8000 seems to be good for humans and bigger
        self.min_area = min_area

    def detect(self):
        # initialize variables
        frame_counter = 0
        motion_frames = []
        first_frame = None

        # loop over the frames of the video
        while self.vidcap.isOpened():
            # grab the current frame
            frame = self.vidcap.read()[1]
            motion = False

            # if the frame could not be grabbed, then we have reached the end
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
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < self.min_area:
                    continue

                # compute the bounding box for the contour, draw it on the frame
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion = True

            if motion:
                motion_frames.append(frame_counter)

            # Increment frame counter
            frame_counter += 1

        self.vidcap.release()
        return motion_frames
