import enum
import json
import os
import glob
import sys

import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
import mediapipe as mp
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Flatten, Dense
import generate_pixel_map

import livelossplot
from tensorflow.python.keras.optimizer_v2.adam import Adam

plot_losses = livelossplot.PlotLossesKeras()

from tensorflow.python.keras.callbacks import ModelCheckpoint



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

mp_holistic = mp.solutions.holistic


class PoseLandmark(enum.IntEnum):
    """The 25 (upper-body) pose landmarks."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS = frozenset([
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER),
    (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR),
    (PoseLandmark.MOUTH_RIGHT, PoseLandmark.MOUTH_LEFT),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB),
    (PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB),
    (PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_HIP)
])


def get_pose(file):
    pose = mp_pose.Pose(upper_body_only=True, static_image_mode=False, smooth_landmarks=False,
                        min_detection_confidence=0.4, min_tracking_confidence=0.4)

    cap = cv2.VideoCapture(file)
    # cap.set(1, 2)

    landmarks = []

    print(file)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print('========================================')
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, POSE_CONNECTIONS)
        # cv2.imshow('MediaPipe Pose', image)

        # print(results.pose_landmarks.landmark[0])
        # print(results.pose_landmarks.landmark[24])

        landmark = []

        for i in range(0, 25):  # 25 first keypoint, upper body only
            landmark.append(position_2_array(results.pose_landmarks.landmark[i]))
            # print(position_2_array(results.pose_landmarks.landmark[i]))

        # print(landmark)
        landmarks.append(landmark)

        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
    cap.release()
    # print(landmarks)
    return landmarks


def annotate_dir(dir):
    print("----------------------")
    folders = [f.path for f in os.scandir(dir) if f.is_dir()]
    print(folders)

    print(dir)

    videoPaths = glob.glob(dir + "/*")

    for vid in videoPaths:
        try:

            print(vid[:-3] + "txt")

            landmark = get_pose(vid)

            text_file = open(vid[:-3] + "txt", "w")
            text_file.write(str(landmark))
            text_file.close()

        except:
            False


def position_2_array(position):
    # list_pos = [position.x, position.y, position.z]
    list_pos = [position.x, position.y]
    return list_pos


def read_txt():
    f = open(r"C:\Users\minelab\dev\TSL\cropped_video\P01_s1_00_0.txt")
    text = f.read()
    res = json.loads(text)

    array_np = np.array(res)

    print(array_np)
    print(array_np.shape)


import tensorflow as tf

RESOLUTION = 224
OUTPUT = 56
DATASET_PATH = r'G:\TSL\temp\Cropped-Video'
# DATASET_PATH = r'G:\TSL\temp\video-cropped'
# DATASET_PATH = r'G:\TSL\temp\video-croppe2'
CHANNEL = 3
EXT_POS = 38
EPOCH = 10


def train_model():
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # config = tf.ConfigProto(device_count={'GPU': 0})
    inputs = tf.keras.Input(name="input_1", shape=(RESOLUTION, RESOLUTION, CHANNEL))

    x = Conv2D(name="block1_conv1", input_shape=(RESOLUTION, RESOLUTION, CHANNEL), filters=64, kernel_size=(3, 3),
               padding="same",
               activation="relu")(inputs)
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
    # x = Conv2D(filters=CHANNEL, kernel_size=(3, 3), padding="same", activation="relu")(x)
    #
    x = UpSampling2D(size=(2, 2,))(x)
    # x = UpSampling2D(size=(2, 2,))(x)
    # Conv2D(CHANNEL, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

    # Keypoint
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(50, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    # model.compile(optimizer="Adam", loss="mse", metrics=["mae", "accuracy"])

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss="mse", metrics=['accuracy', 'mae'])
    # model.compile(optimizer='sgd', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy', 'mae'])

    model.summary()

    X_data = generate_data_x(DATASET_PATH)
    Y_data = generate_keypoint_data(DATASET_PATH)

    # X_data = generate_data_x(DATASET_PATH)
    # Y_data = generate_pixel_data(DATASET_PATH)

    # Y_data = np.reshape(Y_data, (X_data.shape[0], RESOLUTION * RESOLUTION))
    Y_data = np.reshape(Y_data, (X_data.shape[0], 50))

    print(X_data.shape)
    print(Y_data.shape)

    # # Convert data keypoint ONLY FOR KEYPOINT
    # Y_data = Y_data * RESOLUTION

    # # checkpoint
    # filepath = "weights-improvement-{epoch:02d}-{loss:.2f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    # callbacks_list = [checkpoint]
    #
    # model.fit(x=X_data, y=Y_data, batch_size=8, epochs=20, callbacks=callbacks_list)

    model.fit(x=X_data, y=Y_data, batch_size=256, epochs=EPOCH, callbacks=[plot_losses], validation_split=0.2,
              shuffle=True)

    # model.save(filepath=r'C:\Users\minelab\dev\TSL\model\model_using_pixel.h5')
    model.save(filepath=r'C:\Users\minelab\dev\TSL\model\model_full_data_XXX.h5')

    # model = tf.keras.Model(inputs=inputs, outputs=outputs)


def generate_data_x(dir):
    folders = [f.path for f in os.scandir(dir) if f.is_dir()]
    print(folders)

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    print(onlyfiles)

    filePaths = glob.glob(dir + "/*")

    data = []

    for file in filePaths:

        # print(file[EXT_POS:])

        if file[EXT_POS:] == 'avi':
            try:
                cap = cv2.VideoCapture(file)

                # cap.set(1, 2)

                while cap.isOpened():
                    ret, frame = cap.read()

                    # (height, width) = frame.shape[:2]
                    if ret:

                        resized_image = cv2.resize(frame, (RESOLUTION, RESOLUTION))
                        # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                        #
                        # back2rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                        # data.append(resized_image)

                        # cv2.imshow('1',resized_image)
                        # cv2.imshow('2',np.fliplr(resized_image))
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                        # data.append(resized_image)
                        data.append(np.fliplr(resized_image))
                        # cv2.imshow('vid', frame)
                        #
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
                    else:
                        break

                cap.release()

            except cv2.error as e:
                print(e)
                False

    data_np = np.array(data)

    print(data_np.shape)

    return data_np


def generate_keypoint_data(dir):
    folders = [f.path for f in os.scandir(dir) if f.is_dir()]
    print(folders)

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    print(onlyfiles)

    filePaths = glob.glob(dir + "/*")

    data = []

    frames = []

    for file in filePaths:
        if file[EXT_POS:] == 'txt':
            f = open(file)
            text = f.read()
            res = json.loads(text)

            for i in range(0, len(res)):
                poses = []
                for j in range(0, len(res[i])):
                    point = [res[i][j][0], res[i][j][1]]
                    poses.append(np.asarray(point))
                frames.append(np.asarray(poses))
                # print(np.asarray(frames).shape)
                # data.append(frames)
            # np_res = np.array(frames)
            # print(np_res.shape)
            # data.append(frames)
            f.close()

    data_np = np.array(frames)
    #
    print(data_np.shape)
    return data_np


def generate_pixel_data(dir):
    folders = [f.path for f in os.scandir(dir) if f.is_dir()]
    print(folders)

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    print(onlyfiles)

    filePaths = glob.glob(dir + "/*")

    data = []

    frames = []

    for file in filePaths:
        if file[EXT_POS:] == 'txt':
            f = open(file)
            text = f.read()
            res = json.loads(text)

            for i in range(0, len(res)):
                pixel = generate_pixel_map.generate(res[i], OUTPUT, OUTPUT, write_image=False)
                # cv2.imshow(file, pixel)
                # cv2.waitKey(10)
                # jangan lupa di flip imagenya
                frames.append(np.asarray(pixel))

            # np_res = np.array(frames)
            # print(np_res.shape)
            # data.append(frames)
            f.close()

    data_np = np.array(frames)

    print(data_np.shape)
    return data_np


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


if __name__ == '__main__':
    # get_pose(r'C:\Users\minelab\dev\TSL\cropped_video\P01_s1_00_0.avi')
    # annotate_dir(r'C:\Users\minelab\dev\TSL\cropped_video')
    # annotate_dir(r'G:\TSL\temp\Cropped-Video')
    # read_txt()

    train_model()

    # generate_data_x(r'C:\Users\minelab\dev\TSL\cropped_video')
    # generate_data_x(r'G:\TSL\temp\Cropped-Video')
    # generate_keypoint_data(r'C:\Users\minelab\dev\TSL\cropped_video')
    #
    # generate_keypoint_data(r'G:\TSL\temp\Cropped-Video')
    # generate_pixel_data(r'G:\TSL\temp\Cropped-Video')

    # Xdata = generate_data_x(r'G:\TSL\temp\video-cropped')
    # Ydata = generate_pixel_data(r'G:\TSL\temp\video-cropped')
    # #
    # visualize_dataset(Xdata, Ydata)
