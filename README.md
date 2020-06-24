# deepfake-detection-challenge-dataset-practice
Uses BlazeFace for face detection using pretrained model from Google's MediaPipe framework with some changes that can be found at https://github.com/hollance/BlazeFace-PyTorch

Additionally, the starter kit comes from a Kaggle notebook at https://www.kaggle.com/gpreda/deepfake-starter-kit

Requires python, numpy, PyTorch, pandas, OpenCV, matplotlib, dlib

For this repository to work, video files need to be downloaded from https://www.kaggle.com/gpreda/deepfake-starter-kit/#data

Some screenshots of each stage:

Blazeface face detection with feature keypoints:

![alt text](https://github.com/jklewis99/deepfake-detection-challenge-dataset-practice/blob/master/screenshots/blazeface_face1.png?raw=true)
![alt text](https://github.com/jklewis99/deepfake-detection-challenge-dataset-practice/blob/master/screenshots/blazeface_face2.png?raw=true)
![alt text](https://github.com/jklewis99/deepfake-detection-challenge-dataset-practice/blob/master/screenshots/blazeface_face3.png?raw=true)

Arithmetic resize with dlib 68 feature keypoints:

![alt text](https://github.com/jklewis99/deepfake-detection-challenge-dataset-practice/blob/master/screenshots/resize_and_landmark_face1.png?raw=true)
![alt text](https://github.com/jklewis99/deepfake-detection-challenge-dataset-practice/blob/master/screenshots/resize_and_landmark_face2.png?raw=true)
![alt text](https://github.com/jklewis99/deepfake-detection-challenge-dataset-practice/blob/master/screenshots/resize_and_landmark_face3.png?raw=true)
