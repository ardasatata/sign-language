import glob
import json
import os
from os import listdir
from os.path import isfile, join

import gc

import livelossplot
import tensorflow as tf
from keras.preprocessing import sequence
from tensorflow import keras

from tensorflow.keras.models import Model
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Add, Activation, Lambda, Dense, TimeDistributed, Conv2D, \
    MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM, Dropout, MaxPool2D, UpSampling2D
from tensorflow.python.keras.optimizer_v1 import Adam

import generate_pixel_map
from extract_layer4 import get_output_layer
from train_custom import VGG, TCN_layer

from keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
plot_losses = livelossplot.PlotLossesKeras()

from PIL import Image
from numpy import savez_compressed
import numpy as np
import cv2
import random

import mediapipe as mp

from tensorflow.keras.utils import Progbar
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

MODEL = r"C:\Users\minelab\dev\TSL\model\model_3_point_full_data.h5"

MODEL_SAVE_PATH = r'D:\TSL\\'

DIR = r'G:\TSL\Processed-Dataset\Color-Cropped'
DIR_CROPPED_KEY = r'G:\TSL\temp\Cropped-Video'
SAVE_PATH = r'D:\LSA64\LSA64_Cropped\\'
NPZ_DIR = r'D:\LSA64\LSA64_10Class_VGGOutput'

START_FROM = 0

# 10 Class with full sample & subject
TOTAL_CLASS = 10
TOTAL_SUBJECT = 50
TOTAL_SAMPLE = 5

# # Testing
# TOTAL_CLASS = 2
# TOTAL_SUBJECT = 2
# TOTAL_SAMPLE = 2

RESOLUTION = 112
OUTPUT = 56
CHANNEL = 3

FILENAME_PADDING = 15

EXT = r'mp4'

MAX_FRAME = 140

EXTRACT_DESTINATION = r'D:\LSA64\LSA64_10Class_VGGOutput\\'

EPOCH = 20

S_CONST = r's1'


def generate_data_list():
    for class_num in range(START_FROM, TOTAL_CLASS):
        for subject_num in range(0, TOTAL_SUBJECT):
            for sample_num in range(0, TOTAL_SAMPLE):
                file = r"{dir}\{class_num}\P{subject_num}_{s_const}_{class_num2}_{sample_num}.{ext}" \
                    .format(dir=DIR, class_num=str(class_num).zfill(6), class_num2=str(class_num).zfill(2),
                            subject_num=str(subject_num + 1).zfill(2), sample_num=str(sample_num),
                            ext='avi', s_const=S_CONST)
                print(file)


CLASS_CROPPED = 3


def generate_data_processed(debug=False):
    videos_array = []
    keypoints_array = []

    for subject_num in range(0, TOTAL_SUBJECT):
        for class_num in range(0, CLASS_CROPPED):
            for sample_num in range(0, TOTAL_SAMPLE):
                video = r"{dir}\P{subject_num}_{s_const}_{class_num}_{sample_num}.{ext}" \
                    .format(dir=DIR_CROPPED_KEY, subject_num=str(subject_num + 1).zfill(2),
                            class_num=str(class_num).zfill(2),
                            sample_num=str(sample_num), ext='avi', s_const=S_CONST)
                keypoint = r"{dir}\P{subject_num}_{s_const}_{class_num}_{sample_num}.{ext}" \
                    .format(dir=DIR_CROPPED_KEY, subject_num=str(subject_num + 1).zfill(2),
                            class_num=str(class_num).zfill(2),
                            sample_num=str(sample_num), ext='txt', s_const=S_CONST)

                keypoints_array.append(keypoint)
                videos_array.append(video)
                if debug:
                    print(video)
                    print(keypoint)

    return videos_array, keypoints_array


def visualize_dataset(Xdata, Ydata):
    cv2.namedWindow('X', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Y', cv2.WINDOW_NORMAL)

    for i in range(0, Xdata.shape[0]):
        # print(Xdata[i])
        # print(Ydata[i])

        cv2.imshow('X', Xdata[i])
        cv2.imshow('Y', Ydata[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # exit(0)

    # print(Xdata)
    # print(Ydata)


def construct_data(videos=[''], keypoints=[''], pixel_map=True, flip=True, preview=False,
                   normalized_pixel=True, full_keypoint=False, debug=False):
    videos_array = []
    keypoints_array = []

    count = 0

    for file in videos:
        # process video
        try:
            cap = cv2.VideoCapture(file)
            # cap.set(1, 2)
            while cap.isOpened():
                ret, frame = cap.read()
                # (height, width) = frame.shape[:2]
                if ret:
                    resized_frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
                    # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                    # back2rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    # data.append(resized_image)

                    if flip:
                        videos_array.append(np.fliplr(resized_frame))
                    else:
                        videos_array.append(resized_frame)

                    if preview:
                        cv2.imshow('vid_preview', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    break
            cap.release()
        except cv2.error as e:
            print(e)
            False

        try:
            f = open(keypoints[count])
            text = f.read()
            res = json.loads(text)

            for i in range(0, len(res)):
                if pixel_map:
                    pixel = generate_pixel_map.generate(res[i], OUTPUT, OUTPUT,
                                                        write_image=False, normalized=normalized_pixel,
                                                        full_keypoint=full_keypoint)
                    keypoints_array.append(np.asarray(pixel))
                else:
                    poses = []
                    for j in range(0, len(res[i])):
                        point = [res[i][j][0], res[i][j][1]]
                        poses.append(np.asarray(point))
                    keypoints_array.append(np.asarray(poses))

            # np_res = np.array(frames)
            # print(np_res.shape)
            # data.append(frames)
            f.close()
        except f.errors as e:
            print(e)
            False

        count += 1

    videos_array = np.array(videos_array)
    keypoints_array = np.array(keypoints_array)
    if debug:
        print(videos_array.shape)
        print(keypoints_array.shape)

    return videos_array, keypoints_array


def visualize_dataset(Xdata, Ydata):
    cv2.namedWindow('X', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Y', cv2.WINDOW_NORMAL)

    for i in range(0, Xdata.shape[0]):

        cv2.imshow('X', Xdata[i])
        cv2.imshow('Y', Ydata[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # exit(0)


def vgg_mod(inputs):
    x = Conv2D(name="block1_conv1", input_shape=(RESOLUTION, RESOLUTION, CHANNEL), filters=64, kernel_size=(3, 3),
               padding="same", activation="relu")(inputs)
    x = Conv2D(name="block1_conv2", filters=64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(name="block1_pool", pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(name="block2_conv1", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(name="block2_conv2", filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(name="block2_pool", pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(name="block3_conv1", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(name="block3_conv2", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(name="block3_conv3", filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)

    # # full vgg 16
    # x = MaxPool2D(name="block3_pool", pool_size=(2, 2), strides=(2, 2))(x)
    # x = Conv2D(name="block4_conv1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    # x = Conv2D(name="block4_conv2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    # x = Conv2D(name="block4_conv3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    # x = MaxPool2D(name="block4_pool", pool_size=(2, 2), strides=(2, 2))(x)
    # x = Conv2D(name="block5_conv1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    # x = Conv2D(name="block5_conv2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    # x = Conv2D(name="block5_conv3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    # x = MaxPool2D(name="block5_pool", pool_size=(2, 2), strides=(2, 2))(x)

    # # Pixel Map
    x = Conv2D(filters=CHANNEL, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = UpSampling2D(size=(2, 2,))(x)

    # x = UpSampling2D(size=(2, 2,))(x)
    # Conv2D(CHANNEL, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

    # # Keypoint
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(50, activation='relu')(x)

    return x


def train():
    # Initial Setup
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # config = tf.ConfigProto(device_count={'GPU': 0})
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    inputs = tf.keras.Input(name="input_1", shape=(RESOLUTION, RESOLUTION, CHANNEL))

    # Contruct Model
    x = vgg_mod(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.summary()

    # Define Loss,Optimizer,Metrics
    opt = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MAE
    metrics_names = ['accuracy', 'mae']

    model.compile(optimizer=opt, loss=loss, metrics=metrics_names)

    x_list, y_list = generate_data_processed(False)

    print(np.asarray(x_list).shape)
    print(np.asarray(y_list).shape)

    batch_size = 2
    epochs = EPOCH

    loss_ = 999999999

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch}')

        mae = []
        acc = []
        loss = []

        pb_i = Progbar(CLASS_CROPPED * TOTAL_SUBJECT * TOTAL_SAMPLE, stateful_metrics=metrics_names)

        for i in range(0, len(x_list) // batch_size):
            X = x_list[i * batch_size:min(len(x_list), (i + 1) * batch_size)]
            Y = y_list[i * batch_size:min(len(x_list), (i + 1) * batch_size)]

            x_data, y_data = construct_data(X, Y, preview=False, normalized_pixel=True, full_keypoint=False)

            # visualize_dataset(x_data, y_data)

            res = model.train_on_batch(np.array(x_data), np.array(y_data))

            values = [('loss', res[0]), ('acc', res[1]), ('mae', res[2])]

            pb_i.add(batch_size, values=values)

            mae.append(res[2])
            acc.append(res[1])
            loss.append(res[0])

        print(f'Loss : {np.average(np.array(loss))}, '
              f'Accuracy : {np.average(np.array(acc))}, '
              f'MAE : {np.average(np.array(mae))}')

        gc.collect()

        if np.average(np.array(loss)) < loss_:
            loss_ = np.average(np.array(loss))
            model.save(filepath=f'{MODEL_SAVE_PATH}{"model"}{loss_}.h5')

        # val_loss = []
        # for i in range(0, len(x_data) // batch_size):
        #     X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
        #     y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]
        #     val_loss.append(tcn_model.validate_on_batch(X, y))
        #
        # print('Validation Loss: ' + str(np.mean(val_loss)))

    model.save(filepath=f'{MODEL_SAVE_PATH}jambrol.h5')


if __name__ == '__main__':
    # generate_data_list()
    # generate_data_processed()

    # train()

    testVideo = [r'G:\TSL\temp\Cropped-Video\P06_s1_00_1.avi', r'G:\TSL\temp\Cropped-Video\P06_s1_00_2.avi']
    testKeypoint = [r'G:\TSL\temp\Cropped-Video\P06_s1_00_1.txt', r'G:\TSL\temp\Cropped-Video\P06_s1_00_2.txt']
    #
    x_data, y_data = construct_data(testVideo, testKeypoint, preview=False, normalized_pixel=False)
    #
    visualize_dataset(x_data, y_data)
