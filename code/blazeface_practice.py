import numpy as np
import torch
import os
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blazeface import BlazeFace
import math
import dlib

DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'
PREDICTOR = dlib.shape_predictor('../input/shape_predictor_68_face_landmarks.dat')

def get_frame_faces(video_path, net):
    '''
    input: video_path - path for video
    process:
    1. perform a video capture from the video
    2. read the image
    3. detect faces in the image
    '''
    capture_image = cv.VideoCapture(video_path)
    success, orig_frame = capture_image.read()
    orig_frame = cv.cvtColor(orig_frame, cv.COLOR_BGR2RGB)
    # for blazeface, input image needs to be 128 X 128
    scale_value = 128/min(orig_frame.shape[0], orig_frame.shape[1])
    
    dim = (int(orig_frame.shape[1] * scale_value),
           int(orig_frame.shape[0] * scale_value))
    # we want to maintain as much aspect ratio as possible
    frame = cv.resize(orig_frame, dim, interpolation=cv.INTER_AREA)
    # get the image to 128x128
    frame, cropped_dims = crop_to_square(frame, 128)
    cv.waitKey(0)
    detections = net.predict_on_image(frame)
    faces_endpoints = plot_detections(frame, detections) # will return an array of face endpoints
    # perform some simple arithmetic to draw the face box on the original image with higher resolution
    plot_original(orig_frame, scale_value, faces_endpoints, cropped_dims)

def plot_original(img, scale_value, faces_endpoints, cropped_dims):
    '''
    img - the original image on which we want to find the face
    scale_value - the ORIGINAL values used to rescale the image with respect
                  to the minimum dimension
    rectangle_dims - array with tuples of shape (ymin, xmin, ymax, xmax) which mark
                     the rectangle box dimensions of the face detected in the 
                     128x128 image
    cropped_dims - (y, x) the amount taken from the top and left, respectively
                    when it was cropped to 128x128
    '''
    fig, ax=plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    print('Face endpoints on original image:')
    for face in faces_endpoints:
        ymin = (face[0] + cropped_dims[0]) * 1 / scale_value
        xmin = (face[1] + cropped_dims[1]) * 1 / scale_value
        ymax = (face[2] + cropped_dims[0]) * 1 / scale_value
        xmax = (face[3] + cropped_dims[1]) * 1 / scale_value

        print('Y: ({}, {}); X: ({}, {})'.format(ymin, ymax, xmin, xmax))
        rect=patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    linewidth = 1, edgecolor = "r", facecolor = "none")
                                    # alpha = detections[i, 16])
        ax.add_patch(rect)
        get_landmarks(img, dlib.rectangle(int(xmin), int(ymin), int(xmax), int(ymax)), ax)
    plt.show()

def crop_to_square(img, size):
    '''
    img - the numpy array that we wish to resize
    size - the width of the square we need to produce
    '''
    pxls_to_crop_x = (img.shape[1] - size) / 2
    pxls_to_crop_y = (img.shape[0] - size) / 2
    # have to use ceilings and floors to ensure size variable is the final height and width dimensions
    cropped_img = img[int(math.floor(pxls_to_crop_y)) : img.shape[0] - int(math.ceil(pxls_to_crop_y)), int(math.floor(pxls_to_crop_x)) : img.shape[1] - int(math.ceil(pxls_to_crop_x))]
    return cropped_img, (pxls_to_crop_y, pxls_to_crop_x)

def plot_detections(img, detections, with_keypoints=True):
    fig, ax=plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)

    if isinstance(detections, torch.Tensor):
        detections=detections.cpu().numpy()

    if detections.ndim == 1:
        detections=np.expand_dims(detections, axis=0)

    print("Found %d face(s)" % detections.shape[0])
    print('Face endpoints on 128x128 image:')
    detected_faces_endpoints = []

    for i in range(detections.shape[0]):
        ymin=detections[i, 0] * img.shape[0]
        xmin=detections[i, 1] * img.shape[1]
        ymax=detections[i, 2] * img.shape[0]
        xmax=detections[i, 3] * img.shape[1]

        detected_faces_endpoints.append((ymin, xmin, ymax, xmax))
        print('Y: ({}, {}); X: ({}, {})'.format(ymin, ymax, xmin, xmax))
        rect=patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth = 1, edgecolor = "r", facecolor = "none",
                                 alpha = detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x=detections[i, 4 + k*2] * img.shape[1]
                kp_y=detections[i, 4 + k*2 + 1] * img.shape[0]
                circle=patches.Circle((kp_x, kp_y), radius = 0.5, linewidth = 1,
                                        edgecolor = "lightskyblue", facecolor = "none",
                                        alpha = detections[i, 16])
                ax.add_patch(circle)

    plt.show()
    return detected_faces_endpoints

def get_landmarks(img, face, ax):
    landmarks = PREDICTOR(img, face)
    # dlib landmarks will extract 68 landmarks.
    for landmark_idx in range(68):
        x, y = landmarks.part(landmark_idx).x, landmarks.part(landmark_idx).y
        lm_circle = patches.Circle((x, y), radius = 1, linewidth = 2, edgecolor = "lightskyblue", facecolor = "none")
        ax.add_patch(lm_circle)

def get_meta_from_json(path, json_file):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

def main():
    # Here we check the train data files extensions.
    train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
    ext_dict = []
    for file in train_list:
        file_ext = file.split('.')[1]
        if (file_ext not in ext_dict):
            ext_dict.append(file_ext)
    print(f"Extensions: {ext_dict}")

    # Let's count how many files with each extensions there are.
    for file_ext in ext_dict:
        print(
            f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")

    test_list = list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))
    ext_dict = []
    for file in test_list:
        file_ext = file.split('.')[1]
        if (file_ext not in ext_dict):
            ext_dict.append(file_ext)
    print(f"Extensions: {ext_dict}")
    for file_ext in ext_dict:
        print(
            f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")

    json_file = [file for file in train_list if file.endswith('json')][0]
    print(f"JSON file: {json_file}")

    meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER, json_file)
    meta_train_df.head()

    fake_train_sample_video = list(
        meta_train_df.loc[meta_train_df.label == 'FAKE'].sample(3).index)
    real_train_sample_video = list(
        meta_train_df.loc[meta_train_df.label == 'REAL'].sample(3).index)

    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())

    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(gpu)
    
    net=BlazeFace().to(gpu)
    net.load_weights("../input/blazeface.pth")
    net.load_anchors("../input/anchors.npy")

    for video_file in fake_train_sample_video:
        get_frame_faces(os.path.join(
            DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file), net)
    # get_frame_faces()
    # Optionally change the thresholds:
    # net.min_score_thresh = 0.75
    # net.min_suppression_threshold = 0.3

if __name__ == '__main__':
    main()