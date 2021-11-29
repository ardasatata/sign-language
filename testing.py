import tensorflow as tf

import keras

# print(tf.version.VERSION)
# print(keras.__version__)
from os.path import isfile, join, isdir

import numpy as np
import cv2
from numpy import savez_compressed
import os
import gc
from extract_layer4 import get_output_layer

# x_npz = np.load(r"F:\Dataset\Sign Language\Demo CSL\Temp\11_05_2021_10_48_14.npz", mmap_mode='r')
#
# print(x_npz['arr_0'])
# print(np.asarray(x_npz['arr_0']).shape)

CROP_X = 200
CROP_TOP = 200

PREVIEW = False
DEBUG = False


def crop_video(file, fileName):
    if DEBUG:
        print(file)
        print(fileName)

    save_file = f'F:\Dataset\Sign Language\Output-Test\{fileName}.npz'

    if isfile(save_file):
        print('exist ' + save_file)
        return

    try:
        cap = cv2.VideoCapture(file)
        video = []

        while cap.isOpened():
            ret, frame = cap.read()
            # (height, width) = frame.shape[:2]
            if ret:

                if PREVIEW:
                    cv2.imshow('orig', frame)

                cropped = frame[0 + CROP_TOP:720, 0 + CROP_X:1280 - CROP_X]

                if PREVIEW:
                    cv2.imshow('cropped', cropped)

                resized_image = cv2.resize(cropped, (224, 224))

                if PREVIEW:
                    cv2.imshow('resized', resized_image)

                # append frame to be converted
                video.append(np.asarray(resized_image))

                if PREVIEW:
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            else:
                break

        # cv2.waitKey(125)

        if DEBUG:
            print(np.asarray(video).shape)

        output = get_output_layer(src=np.asarray(video))
        tf.keras.backend.clear_session()
        gc.collect()

        if DEBUG:
            print(output.shape)

        save_dir = f'F:\Dataset\Sign Language\Output-Test'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        savez_compressed(save_file, output)

        # exit(0)

        cap.release()

    except cv2.error as e:
        print(e)
        False

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    crop_video(file=r"F:\Dataset\Sign Language\CSL\pytorch\color\000000\P01_s1_00_0._color.avi", fileName="conv5_block3_1_conv")
