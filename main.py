# Measure-Command { py main.py | Out-Host }
# Usage :: python3 main.py -v "videos/truck.mp4" -m "lenient" -o "output/" -f -d
import cv2
import os
import argparse
import pathlib
import json
from sys import platform
from math import floor
from shutil import rmtree

# Don't import if running on Windows
if platform != 'win32':
    from face_rec import FaceRec
from motion_detection import MotionDetection
from colour_detection import ColourDetection
from object_detection import ObjectDetection


def check_video_validity(video):
    video_file_extensions = ['flv', 'm4v', 'mp4', 'wmv', 'avi', 'mkv']
    vidcap = cv2.VideoCapture(video)
    ret, frame = vidcap.read()
    if not ret:
        print('File could not be found (or may be corrupted). Please check spelling.')
        exit(0)
    else:
        ext = video.split('.')[-1]
        if ext not in video_file_extensions:
            print('Please use one of the following video formats:')
            print(video_file_extensions)
            exit(0)


def get_filename(path):
    return path.split('/')[-1]


def get_video_fps(video):
    vidcap = cv2.VideoCapture(video)
    return vidcap.get(cv2.CAP_PROP_FPS)


def get_video_length(video):
    vidcap = cv2.VideoCapture(video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count / fps


# FIXME: This -> [8, 16, 32, 48, 80, 328, 344, 352, 416] (from facial recognition)
# FIXME: Returned this -> [["0.3", "0.5"], ["1.1", "1.6"], ["2.7", "10.9"], ["11.5", "11.7"], ["-0.0", "13.9"]]
# FIXME: It did this because there were an uneven number of frames
def get_timestamps(frames, video):
    # Returns array of tuples (start_time, end_time)
    fps = get_video_fps(video)
    timestamps_local = []
    start = -1
    for i in range(len(frames)):
        if i != len(frames) - 1:
            frame_num = frames[i]
            next_frame_num = frames[i+1]
            if start == -1:
                start = frame_num
            elif frame_num == next_frame_num - 1:
                continue
            else:
                timestamps_local.append(('%.1f' % (start/fps), '%.1f' % (frame_num/fps)))
                start = -1
        else:
            timestamps_local.append(('%.1f' % (start/fps), '%.1f' % (frames[i]/fps)))
    return timestamps_local


# Source: https://stackoverflow.com/questions/30136257/how-to-get-image-from-video-using-opencv-python
def video_to_frames(video, path_frames_dir):
    # extract frames from a video and save to directory as 'x.jpg' where x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            # This needs to be .jpg for the face_rec to load image properly
            cv2.imwrite(os.path.join(path_frames_dir, '%d.jpg') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


def find_common_frames(frames1, frames2):
    return list(set(frames1) & set(frames2))


# TODO add option to keep the frames folder in case they want to run it again on same video, will save time
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find anomalitic frames from within a video file')
    parser.add_argument('-v', '--video', help='Location of video file to parse', required=True)
    parser.add_argument('-m', '--mode', help='Determines what mode to run the program under.',
                        choices=['strict', 'colour', 'motion', 'lenient'], default='lenient')
    parser.add_argument('-o', '--output', help='Specify the path to which any output of the program should be saved',
                        default='output/')
    parser.add_argument('-f', '--face_rec',
                        help='Enables facial recognition components if accuracy/features are wanted over '
                             'performance/speed.', action='store_true')
    parser.add_argument('-d', '--detect_objects',
                        help='Enables object recognition components if accuracy/features are wanted over '
                             'performance/speed.', action='store_true')
    args = parser.parse_args()

    check_video_validity(args.video)
    anom_frames = []
    json_skel = {}
    input_filename = get_filename(args.video)
    frames_dir = f'frames/{input_filename.strip(".mp4")}/'
    if not os.path.exists('frames/'):
        os.mkdir('frames/')
    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)
    frames_exists = os.listdir(frames_dir)
    if len(frames_exists) < 2:
        video_to_frames(args.video, frames_dir)
    # Making sure the output directory ends with a slash to avoid confusion later on
    if args.output[-1] != "/":
        args.output = args.output + '/'
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    json_skel['video_filename'] = input_filename
    json_skel['video_length'] = get_video_length(args.video)
    json_skel['screenshot_directory'] = f'{args.output}object_detections/'
    try:
        f = open(f'{args.output}anomalies.json', 'w+')
    except OSError as e:
        print('The output folder chosen can\'t be written to. Please check the permissions and ensure there is '
              'enough free space.')
        print(e)
        exit(0)

    # TODO Need to add either some control over the (8000 and 20) params, or make it part of the "mode" part
    if args.mode == 'strict':
        # This mode is risky, because if no colour change is flagged,
        # then even if motion caught stuff it won't be returned

        # Run motion
        # TODO Add background subtraction to try and remove some shadow/lighting parts
        md = MotionDetection(args.video, 8000)
        motion_frames = md.detect()

        # Run colour
        cd = ColourDetection(frames_dir, 20)
        colour_frames = cd.get_frames()

        # Only return frames that appear in both
        common_frames = find_common_frames(motion_frames, colour_frames)
        anom_frames = common_frames

        # Get timestamp ranges of anomalitic frames (instead of giving frame numbers, which is kind of useless)
        timestamps = get_timestamps(common_frames, args.video)

        # Add to JSON
        json_skel['timestamps'] = timestamps
        json_skel['common_frames'] = common_frames
        json_skel['motion_detection_frames'] = motion_frames
        json_skel['colour_detection_frames'] = colour_frames

    elif args.mode == 'colour':
        # Run colour
        cd = ColourDetection(frames_dir, 20)
        colour_frames = cd.get_frames()
        anom_frames = colour_frames

        # Get timestamp ranges of anomalitic frames (instead of giving frame numbers, which is kind of useless)
        timestamps = get_timestamps(colour_frames, args.video)

        # Add to JSON
        json_skel['timestamps'] = timestamps
        json_skel['colour_detection_frames'] = colour_frames

    elif args.mode == 'motion':
        # Run motion
        md = MotionDetection(args.video, 8000)
        motion_frames = md.detect()
        anom_frames = motion_frames

        # Get timestamp ranges of anomalitic frames (instead of giving frame numbers, which is kind of useless)
        timestamps = get_timestamps(motion_frames, args.video)

        # Add to JSON
        json_skel['timestamps'] = timestamps
        json_skel['motion_detection_frames'] = motion_frames

    elif args.mode == 'lenient':
        # Run motion
        md = MotionDetection(args.video, 8000)
        motion_frames = md.detect()

        # Run colour
        cd = ColourDetection(frames_dir, 20)
        colour_frames = cd.get_frames()

        # Return all frames
        anomaly_frames = list(set(motion_frames+colour_frames))
        anom_frames = anomaly_frames

        # Get timestamp ranges of anomalitic frames (instead of giving frame numbers, which is kind of useless)
        timestamps = get_timestamps(anomaly_frames, args.video)

        # Add to JSON
        json_skel['timestamps'] = timestamps
        json_skel['motion_detection_frames'] = motion_frames
        json_skel['colour_detection_frames'] = colour_frames

    if args.face_rec:
        # Returns [(Bool{Whether known face or not}, frame_number)]
        face_frame_anomaly = []
        known_faces_frames = []
        fr = FaceRec(frames_dir)
        res = fr.scan()
        for fr_res in res:
            face_frame_anomaly.append(fr_res[1])
            if fr_res[0] == 'True':
                known_faces_frames.append(fr_res[1])
        good_faces_timestamps = get_timestamps(known_faces_frames, args.video)
        faces_timestamps = get_timestamps(face_frame_anomaly, args.video)

        # Add to JSON
        json_skel['known_faces_timestamps'] = good_faces_timestamps
        json_skel['general_face_timestamps'] = faces_timestamps

    if args.detect_objects:
        output_dir = f'{args.output}object_detections/'
        # Run object detection on the frames returned by motion detection
        # This returns a 4-tuple: (frame#, img_obj, label_array, confidence_array)
        od = ObjectDetection(anom_frames, frames_dir)
        objects = od.detect()
        objects_detected = []
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            rmtree(output_dir)
            os.mkdir(output_dir)

        object_dict = {}
        for obj in objects:
            for item in obj[2]:
                if item in object_dict:
                    object_dict[item].append((obj[0], obj[1]))
                else:
                    object_dict[item] = [(obj[0], obj[1])]
            fps = get_video_fps(args.video)
            objects_detected.append(('%.1fs' % (obj[0]/fps), obj[2], obj[3]))

        for key in object_dict:
            num_frames = len(object_dict[key])
            chosen_frame = floor(num_frames/2)

            # save output
            cv2.imwrite(f'{output_dir}{object_dict[key][chosen_frame][0]}.jpg', object_dict[key][chosen_frame][1])

        # Add to JSON
        json_skel['objects_detected'] = objects_detected

    with open(f'{args.output}anomalies.json', 'w') as outfile:
        json.dump(json_skel, outfile, indent=4)

    # Delete frames folder
    rmtree('frames')
