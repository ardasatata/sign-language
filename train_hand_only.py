import enum
import json
import os
import glob
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Flatten, Dense
import generate_pixel_map
import matplotlib.pyplot as plt

import livelossplot
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow as tf

from train import plot_losses, PoseLandmark

np.set_printoptions(threshold=np.inf)

RESOLUTION = 224
OUTPUT = 112

# DATASET_PATH = r'G:\TSL\temp\Cropped-Video'

# DATASET_PATH = r'G:\TSL\temp\video-cropped'
# DATASET_PATH = r'G:\TSL\temp\video-croppe2'
DATASET_PATH = r'G:\TSL\temp\video-crop325'

CHANNEL = 3
EXT_POS = 38
EPOCH = 20


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


def train_hand():
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

    # Pixel Map
    x = Conv2D(filters=CHANNEL, kernel_size=(3, 3), padding="same", activation="relu")(x)
    #
    x = UpSampling2D(size=(2, 2,))(x)
    # x = UpSampling2D(size=(2, 2,))(x)
    # Conv2D(CHANNEL, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

    # # Keypoint
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(6, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    # model.compile(optimizer="Adam", loss="mse", metrics=["mae", "accuracy"])

    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss="mse", metrics=['accuracy', 'mae'])
    # model.compile(optimizer='sgd', loss="mse", metrics=['accuracy', 'mae'])
    # model.compile(optimizer='sgd', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy', 'mae'])

    model.summary()

    # X_data = generate_data_x(DATASET_PATH)
    # Y_data = generate_hand_key(DATASET_PATH)

    X_data = generate_data_x(DATASET_PATH)
    Y_data = generate_pixel_data(DATASET_PATH)

    # visualize_dataset(X_data, Y_data)

    # Reshape Data for Keypoint
    # Y_data = np.reshape(Y_data, (X_data.shape[0], RESOLUTION * RESOLUTION))
    # Y_data = np.reshape(Y_data, (X_data.shape[0], 6))

    # # Convert data keypoint ONLY FOR KEYPOINT
    # Y_data = Y_data * RESOLUTION

    print(X_data.shape)
    print(Y_data.shape)

    # # checkpoint
    # filepath = "weights-improvement-{epoch:02d}-{loss:.2f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    # callbacks_list = [checkpoint]
    #
    # model.fit(x=X_data, y=Y_data, batch_size=8, epochs=20, callbacks=callbacks_list)
    # model.load_weights(r"C:\Users\minelab\dev\LSA-Dataset\50e_2_.h5", by_name=True)

    model.fit(x=X_data, y=Y_data, batch_size=128, epochs=EPOCH, callbacks=[plot_losses], validation_split=0.2,
              shuffle=True)

    model.save(filepath=r'C:\Users\minelab\dev\TSL\model\model_3_point_PIXEL_325.h5')


def generate_hand_key(dir):
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
                    if j == 0:
                        poses.append(np.asarray(point))
                    if j == 15:
                        poses.append(np.asarray(point))
                    if j == 16:
                        poses.append(np.asarray(point))
                # print(poses)
                # exit(0)
                frames.append(np.asarray(poses))
            f.close()

    data_np = np.array(frames)

    print(data_np.shape)
    return data_np


def generate(points, width, height, img_ref=None, write_image=False):
    data = np.zeros([height, width, 3], dtype=np.uint8)

    # Plot Pixel
    for i in range(0, len(points)):
        # print(round(points[i][0] * width))
        # print(round(points[i][1] * height))
        # data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

        if i == PoseLandmark.LEFT_WRIST:
            data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

        if i == PoseLandmark.RIGHT_WRIST:
            data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

        if i == PoseLandmark.NOSE:
            data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

    normalized_pixel = np.array(generate_pixel_map.NormalizeData(data), dtype='uint8')
    # normalized_pixel = data

    if write_image:
        # Create a figure. Equal aspect so circles look circular
        fig, ax = plt.subplots(1, 2)

        # Show reference image
        if img_ref is not None:
            ref = plt.imread(img_ref)
            ax[1].set_title('Reference Image')
            ax[1].imshow(ref)

        ax[0].set_title('Pixel')
        ax[0].imshow(data)
        cv2.imshow('img original', data)
        cv2.imshow('normalized_pixel', normalized_pixel)
        print(data)
        print(normalized_pixel)
        print(normalized_pixel.shape)
        cv2.imwrite('result_pixel.png', data)
        cv2.imwrite('result_pixel_normalized.png', normalized_pixel)
        cv2.waitKey(10000)  # wait for 50ms
        plt.show()

    return normalized_pixel


def generate_pixel_data(dir):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    print(onlyfiles)

    filePaths = glob.glob(dir + "/*")

    frames = []

    for file in filePaths:
        if file[EXT_POS:] == 'txt':
            f = open(file)
            text = f.read()
            res = json.loads(text)

            for i in range(0, len(res)):
                pixel = generate(res[i], OUTPUT, OUTPUT, write_image=False)
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


if __name__ == '__main__':
    # generate_hand_key(DATASET_PATH)

    # y_ = generate_data_x(DATASET_PATH)
    # x_ = generate_pixel_data(DATASET_PATH)
    #
    # visualize_dataset(x_, y_)

    train_hand()
