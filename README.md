## Requirements
This was run with _Python 3.8.2_  
To install the required dependencies (with the proper, conflict free versions), run: `python3 -m pip install -r requirements.txt`

It is recommended to run the pip install command within a virtual environment so as not to cause conflicts with any existing library versions.

#### Special requirements
The facial recognition library requires additional steps to function properly.

Follow the steps below:
```
git clone https://github.com/davisking/dlib.git

sudo apt-get install build-essential cmake gfortran git wget curl graphicsmagick libgraphicsmagick1-dev libatlas-base-dev libavcodec-dev libavformat-dev libgtk2.0-dev libjpeg-dev liblapack-dev libswscale-dev pkg-config python3-dev python3-numpy software-properties-common zip

cd dlib
mkdir build; cd build; cmake ..; cmake --build .
cd ..
python3 setup.py install (from within venv if desired, if not run as root)
```
If any problems arise, please refer to [this page for guidance](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf).


## Setup
Make sure to have the videos saved in a location that your user has read access.

If you want to create a list of known faces, put clear pictures of the people in <project_root_directory>/known_faces/

Use the name of the person as the filename. If using multiple pictures of 1 person append a number after their name.


## Running the program
The face recognition software doesn't run on Windows (without lots of trouble). So make sure to be using Linux (Ubuntu 20.04 was used in my case).

Run the following command in the root directory (_if using a virtual environment_): `source <venv_folder>/bin/activate`  
#### Modes
The program can be run in 4 different modes (lenient being the default). These modes only affect the combination of motion and colour detection. It does not affect facial recognition or object detection.
1. lenient
    - This mode will run both motion detection and colour detection and will use the combination of their output as the anomalous frames.
2. strict
    - This mode will run both motion detection and colour detection and will use the intersection of their output as the anomalous frames. It is worth noting that if colour change doesn't flag anything then no anomalous frames will be returned.
3. motion
    - This mode will run without the colour change to find the anomalous frames.
4. colour
    - This mode will run without the motion detection to find the anomalous frames.

#### Parameters
1. `-v`, `--video`: Location of video file to parse | Required
2. `-m`, `--mode`: Determines what mode to run the program as | Optional | Default=lenient
3. `-o`, `--output`: Folder in which to save all program output | Optional | Default=.output/
4. `-f`, `--face_rec`: If passed then facial recognition is used | Optional | If omitted facial recognition not use
5. `-d`, `--detect_objects`: If passed then object detection is used | Optional | If omitted object detection not use

#### Examples
This example specifies the walkway.mp4 file to be ingested, uses default mode of lenient, saves the output in a custom directory in their home directory, as well as enable facial recognition and object detection.

`python3 main.py -v "videos/walkway.mp4" -o "~/vdog/walkway_results/" -f -d`

This example specifies the walkway.mp4 file to be ingested, only wants to use motion detection to flag anomalous frames, default output, no facial recognition and no object detection.

`python3 main.py -v "videos/walkway.mp4" -m motion`
## Assumptions
For simplification purposes, some assumptions were made in calculations.

1. The motion detection assumes that the first frame of the video contains no motion in it.
2. Input video files were tested with the `.mp4` extension, however, any of the following should work:
    - `flv`
    - `m4v`
    - `mp4`
    - `wmv`
    - `avi`
    - `mkv`