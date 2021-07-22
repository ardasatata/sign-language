import errno
import gc
import glob
import math
import os
import sys
from os import listdir, path
from os.path import isfile, join, isdir

import cv2
import csv

import re

import progressbar
import tensorflow as tf
from keras.preprocessing import sequence
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Model
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Conv1D, Add, Activation, Lambda, Dense, TimeDistributed, Conv2D, \
    MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM, Dropout, MaxPooling1D, Bidirectional

from extract_layer4 import get_output_layer

import numpy as np
from numpy import savez_compressed

CSL_PATH = r'D:\CSL-25K\pytorch\color'

OUTPUT_PATH = r'D:\CSL-25K\Output Layer 4'

SENTENCE_START = 15
SENTENCE_END = 20

SAMPLE_PER_SENTENCE = 250

PREVIEW = False
DEBUG = False


def load_data():
    folders = [f.path for f in os.scandir(CSL_PATH) if f.is_dir()]

    for sentence in range(SENTENCE_START, SENTENCE_END):
        # print(f'{CSL_PATH}\{str(sentence).zfill(6)}')
        # print(folders[sentence])

        files = [f for f in listdir(folders[sentence]) if isfile(join(folders[sentence], f))]

        with progressbar.ProgressBar(max_value=SAMPLE_PER_SENTENCE) as bar:
            for idx, file in enumerate(files):
                crop_video(f'{folders[sentence]}\{file}', file[:-4], str(sentence).zfill(6))
                bar.update(idx)


CROP_X = 200
CROP_TOP = 200


def crop_video(file, fileName, folderName):
    if DEBUG:
        print(file)
        print(fileName)
        print(folderName)

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

        save_dir = f'{OUTPUT_PATH}\{folderName}'

        # if isdir(save_dir):
        #     print('exist')
        #     exit(0)
        #     savez_compressed(f'{OUTPUT_PATH}\{folderName}\{fileName}.npz', output)
        # else :

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        savez_compressed(f'{OUTPUT_PATH}\{folderName}\{fileName}.npz', output)

        # exit(0)

        cap.release()

    except cv2.error as e:
        print(e)
        False

    tf.keras.backend.clear_session()

    # print('save npz')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load_data() # crop + output 4th layer

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
