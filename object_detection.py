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
