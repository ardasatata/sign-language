import errno
import gc
import glob
import math
import os
import pickle
import sys
from os import listdir, path
from os.path import isfile, join, isdir

from keras_ctcmodel.CTCModel import CTCModel as CTCModel

import cv2
import csv

import re

# import imageio

# import progressbar
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Model
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Conv1D, Add, Activation, Lambda, Dense, TimeDistributed, Conv2D, \
    MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM, Dropout, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import concatenate

from LSA64 import calculate_wer
from extract_layer4 import get_output_layer

import numpy as np
from numpy import savez_compressed

import pandas as pd
import os

from tqdm import tqdm

CSL_PATH = r'F:\Dataset\Sign Language\CSL\pytorch\color'

# OUTPUT_PATH = r'F:\Dataset\Sign Language\CSL-Output'
OUTPUT_PATH = r'F:\Dataset\Sign Language\CSL-Output'
# OUTPUT_PATH = r'F:\Dataset\Sign Language\CSL-Output-ResNet'
KEYPOINT_PATH = r'F:\Dataset\Sign Language\CSL-Key'
MODEL_SAVE_PATH = r"F:\Dataset\Sign Language\Demo CSL\Model Testing"
# MODEL_SAVE_PATH = r"F:\Dataset\Sign Language\CSL Model + Keypoint Resnet"
CLASS_COUNT = 10

SENTENCE_START = 75
SENTENCE_END = 100

SAMPLE_PER_SENTENCE = 250

PREVIEW = False
DEBUG = False
TESTING = False
LOAD_WEIGHT = True

# # TESTING #
# KEYPOINT_PATH = r'F:\Dataset\Sign Language\CSL-Key_test'
# OUTPUT_PATH = r'F:\Dataset\Sign Language\CSL-Output_test'
# MODEL_SAVE_PATH = r"F:\Dataset\Sign Language\CSL Model + Keypoint - Test"
# CLASS_COUNT = 2
# TESTING = True
# # DEBUG = True

# pyshical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(pyshical_devices[0], True)
# tf.config.experimental.set_memory_growth(pyshical_devices[1], True)

selected_joints = {
    '59': np.concatenate((np.arange(0, 17), np.arange(91, 133)), axis=0),  # 59
    '31': np.concatenate((np.arange(0, 11), [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                          [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),  # 31
    '27': np.concatenate(([0, 5, 6, 7, 8, 9, 10],
                          [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                          [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),  # 27
    '27_2': np.concatenate(([0, 5, 6, 7, 8, 9, 10],
                            [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                            [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0)  # 27
}

selected = selected_joints['27']

SENTENCE_LABEL = {0: ['??????', '??????', '???', '??????'], 1: ['???', '??????', '???', '??????', '???', '??????', '???'], 2: ['??????', '??????', '???', '??????'],
                  3: ['???', '??????', '???', '??????', '???', '??????', '???'], 4: ['???', '??????', '???', '??????', '???', '??????'],
                  5: ['???', '??????', '???', '??????', '???', '??????'], 6: ['???', '??????', '???', '??????', '???', '??????'],
                  7: ['???', '??????', '???', '??????', '???', '??????'], 8: ['??????', '??????', '???', '??????'], 9: ['??????', '??????', '???', '??????'],
                  10: ['???', '??????', '???', '??????', '???', '??????'], 11: ['??????', '?????????', '???', '??????'],
                  12: ['???', '??????', '???', '??????', '???', '??????'], 13: ['???', '??????', '???', '??????', '???', '??????'],
                  14: ['???', '??????', '???', '??????', '???', '??????'], 15: ['??????', '??????', '???', '??????', '???'], 16: ['??????', '?????????', '???', '??????'],
                  17: ['??????', '??????', '???', '?????????'], 18: ['???', '??????', '???', '??????', '???', '??????'], 19: ['??????', '??????', '???', '??????'],
                  20: ['???', '??????', '???', '??????'], 21: ['???', '??????', '???', '??????', '???'], 22: ['???', '??????', '???', '???', '???', '???'],
                  23: ['???', '??????', '???', '??????'], 24: ['???', '??????', '???', '??????', '??????'], 25: ['???', '?????????', '???', '???', '???'],
                  26: ['???', '??????', '???', '??????'], 27: ['???', '??????', '???', '??????'], 28: ['???', '?????????', '???', '??????', '???'],
                  29: ['???', '??????', '???', '??????'], 30: ['???', '??????', '???', '???', '???'], 31: ['???', '??????', '???', '??????'],
                  32: ['???', '??????', '???', '???', '???'], 33: ['???', '??????', '???', '??????'], 34: ['???', '??????', '???', '??????'],
                  35: ['???', '?????????', '???', '??????'], 36: ['???', '??????', '???', '??????'], 37: ['???', '??????', '???', '??????'],
                  38: ['???', '??????', '???', '??????', '???'], 39: ['???', '??????', '???', '??????'], 40: ['??????', '???', '??????', '???', '???', '???'],
                  41: ['??????', '??????', '??????', '??????'], 42: ['??????', '???', '??????', '??????'], 43: ['??????', '??????', '??????', '??????'],
                  44: ['??????', '??????', '???', '??????'], 45: ['??????', '??????', '???', '??????'], 46: ['??????', '??????', '???', '??????'],
                  47: ['??????', '??????', '???', '??????', '??????'], 48: ['??????', '??????', '???', '??????'], 49: ['??????', '??????', '???', '??????'],
                  50: ['??????', '???', '??????', '??????', '??????'], 51: ['??????', '???', '??????', '??????'], 52: ['??????', '???', '??????', '??????'],
                  53: ['??????', '??????', '??????', '??????'], 54: ['??????', '???', '??????', '???'], 55: ['??????', '??????', '???', '??????'],
                  56: ['??????', '??????', '???', '??????', '???'], 57: ['??????', '??????', '???', '??????'], 58: ['??????', '??????', '???', '??????', '???'],
                  59: ['??????', '???', '??????', '??????', '???'], 60: ['??????', '???', '??????'], 61: ['??????', '???', '??????'],
                  62: ['??????', '???', '??????', '???', '??????'], 63: ['??????', '???', '??????'], 64: ['??????', '??????', '??????', '???'],
                  65: ['??????', '??????', '??????', '??????'], 66: ['??????', '???', '??????', '??????', '??????'], 67: ['??????', '??????', '???', '??????'],
                  68: ['??????', '???', '??????', '???'], 69: ['??????', '??????', '??????', '???'], 70: ['??????', '??????', '??????', '??????'],
                  71: ['??????', '???', '??????', '???'], 72: ['??????', '??????', '???', '???', '???'], 73: ['??????', '??????', '???', '???'],
                  74: ['??????', '???', '???', '???', '???'], 75: ['??????', '???', '??????', '???'], 76: ['??????', '??????', '???', '??????'],
                  77: ['???', '??????', '??????'], 78: ['??????', '??????', '??????'], 79: ['??????', '??????', '??????'],
                  80: ['??????', '??????', '???', '???', '???'],
                  81: ['??????', '??????', '???'], 82: ['?????????', '???', '???', '???'], 83: ['??????', '???', '??????', '???'],
                  84: ['??????', '??????', '?????????'],
                  85: ['??????', '???', '??????'], 86: ['??????', '???', '??????', '???'], 87: ['??????', '???', '??????'], 88: ['??????', '???', '??????'],
                  89: ['??????', '??????', '???'], 90: ['??????', '??????', '??????'], 91: ['??????', '?????????', '???', '??????'], 92: ['???', '??????', '??????'],
                  93: ['??????', '??????', '??????'], 94: ['???', '??????', '???', '???', '??????'], 95: ['?????????', '???', '??????'],
                  96: ['??????', '???', '??????', '???', '??????', '???'], 97: ['??????', '??????', '???', '??????', '???', '??????'],
                  98: ['??????', '??????', '???', '??????', '???'], 99: ['??????', '??????', '???', '???', '???']}


def load_data():
    folders = [f.path for f in os.scandir(CSL_PATH) if f.is_dir()]

    for sentence in range(SENTENCE_START, SENTENCE_END):
        # print(f'{CSL_PATH}\{str(sentence).zfill(6)}')
        # print(folders[sentence])

        files = [f for f in listdir(folders[sentence]) if isfile(join(folders[sentence], f))]

        with tqdm(total=SAMPLE_PER_SENTENCE) as bar:
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

    save_file = f'{OUTPUT_PATH}\{folderName}\{fileName}.npz'

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

        save_dir = f'{OUTPUT_PATH}\{folderName}'

        # if isdir(save_dir):
        #     print(save_dir)
        #     print('exist')
        #     exit(0)
        #     savez_compressed(f'{OUTPUT_PATH}\{folderName}\{fileName}.npz', output)
        # # else :

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        savez_compressed(save_file, output)

        # exit(0)

        cap.release()

    except cv2.error as e:
        print(e)
        False

    tf.keras.backend.clear_session()

    # print('save npz')


def VGG_2(i_vgg):
    #    input_data = Input(name='input', shape=(None,224, 224, 3), dtype = "float16")
    # Izda.add(TimeDistributed(
    #    Convolution2D(40,3,3,border_mode='same'), input_shape=(sequence_lengths, 1,8,10)))
    #    model = Sequential()
    # Izda.add(TimeDistributed(
    #    Convolution2D(40,3,3,border_mode='same'), input_shape=(sequence_lengths, 1,8,10)))

    # #    i_vgg = tf.keras.layers.Input(batch_shape=(None,55,224,224,3))
    # model = TimeDistributed(
    #     Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name='block1_conv1'))(i_vgg)
    # model = TimeDistributed(
    #     Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name='block1_conv2'))(model)
    # model = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model)
    # model = TimeDistributed(
    #     Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", name='block2_conv1'))(model)
    # model = TimeDistributed(
    #     Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", name='block2_conv2'))(model)
    # model = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model)
    model1 = TimeDistributed(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", name='block3_conv1'))(i_vgg)
    model1 = TimeDistributed(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", name='block3_conv2'))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", name='block3_conv3'))(model1)
    model1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name='block4_conv1'))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name='block4_conv2'))(model1)

    # model1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model1)
    model = TimeDistributed(GlobalAveragePooling2D(name="global_max_full"))(model1)

    #    model = (MaxPooling3D(pool_size=(1, 2, 2)))(model)
    #    model1 = (TimeDistributed(GlobalAveragePooling2D(name="global_max_full")))(model1)

    #    model.compile(loss='mean_squared_error', optimizer='adam')  #,

    return model


# Residual block
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='elu')(
        x)  # first convolution
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # Second convolution
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut (shortcut)
    o = Add()([r, shortcut])
    o = Activation('elu')(o)  # Activation function
    return o


def TCN_layer(input_layer, kernel):
    #    inputs=Input(shape=(28,28))
    # print(input_layer)
    x = ResBlock(input_layer, filters=64, kernel_size=kernel, dilation_rate=1)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=2)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=4)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=8)
    #    x=Flatten()(x)
    return x


# Need to hardcode
# skipped max length = 326

def predict_ctc(file_path="D:\\Dataset\\Sign Language\\CSL-Output/000000/P01_s1_00_0._color.npz"):
    x_data, y_data, x_len, y_len, x_data_keypoint, classes_col, target_data, video_col = generate_data(
        class_count=CLASS_COUNT)

    skipped_max_len = 326
    # 100 total word is 193 words
    num_classes = 193

    print(x_data[0])
    print(y_data[0])
    print(x_len[0])
    print(y_len[0])
    print(x_data_keypoint[0])
    print(classes_col[0])
    print(target_data[0])

    data_index = np.where(x_data == file_path)
    data_index = int(data_index[0])
    print('index = ' + str(data_index))
    print(y_data[data_index])
    print(x_len[data_index])
    print(y_len[data_index])
    print(x_data_keypoint[data_index])
    print(classes_col[data_index])
    print(target_data[data_index])
    print(video_col[data_index])

    # if shuffle:
    #     c = list(zip(x_data, y_data))
    #     random.shuffle(c)
    #     x_data, y_data = zip(*c)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(skipped_max_len, 56, 56, 256))

    # Input Keypoint
    i_keypoint = tf.keras.Input(name="input_1_keypoint", shape=(skipped_max_len, 1, 27, 3))
    output_keypoint = TimeDistributed(GlobalAveragePooling2D(name="global_max_full"))(i_keypoint)
    dense_input_keypoint = Dense(256, activation='relu', name='dense_keypoint')(output_keypoint)

    vgg = VGG_2(i_vgg)
    # m_vgg = Model(inputs=[i_vgg], outputs=[vgg])

    # TCN

    '''
    TCN -> Dense
    '''

    o_tcn_full = TCN_layer(vgg, 5)
    o_tcn_keypoint = TCN_layer(dense_input_keypoint, 5)

    # global_pool = GlobalAveragePooling2D(name="global_max_full")(o_tcn_full)
    # flatten = Flatten()(o_tcn_full) # using flatten to sync the network size - disabled if 'TMC FULL'

    dense = Dense(256, name='dense_o_tcn1')(o_tcn_full)
    dense_keypoint = Dense(256, activation='relu', name='dense_o_keypoint')(o_tcn_keypoint)

    '''
    TMC (cont)
    '''

    o_tcn_block1 = TCN_layer(dense, 1)
    o_tcn_block1 = Dense(256, name='dense_o_tcn_intra_block1')(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    block1 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block1)

    o_tcn_key_block1 = TCN_layer(dense_keypoint, 1)
    o_tcn_key_block1 = Dense(256, name='dense_o_tcn_key_intra_block1')(o_tcn_key_block1)
    o_tcn_key_block1 = Dense(256)(o_tcn_key_block1)
    block1_key = MaxPooling1D(pool_size=5, strides=2)(o_tcn_key_block1)

    i_tcn2 = block1
    o_tcn2 = TCN_layer(i_tcn2, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2 = Dense(256, name='dense_o_tcn2')(o_tcn2)
    o_tcn_block2 = TCN_layer(dense2, 1)
    o_tcn_block2 = Dense(512, name='dense_o_tcn_intra_block2')(o_tcn_block2)
    o_tcn_block2 = Dense(512)(o_tcn_block2)
    block2 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block2)

    i_tcn2_key = block1_key
    o_tcn2_key = TCN_layer(i_tcn2_key, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2_key = Dense(256, activation='relu', name='dense_o_tcn2_key')(o_tcn2_key)
    o_tcn_key_block2 = TCN_layer(dense2_key, 1)
    o_tcn_key_block2 = Dense(256, name='dense_o_tcn_key_intra_block2')(o_tcn_key_block2)
    o_tcn_key_block2 = Dense(256)(o_tcn_key_block2)
    block2_key = MaxPooling1D(pool_size=5, strides=2)(o_tcn_key_block2)

    # Concat VGG + Keypoint
    concat = concatenate([block2, block2_key], axis=2)

    '''
    TMC (cont) # endregion

    '''

    '''
    Sequence Learning
    '''

    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(concat)
    dense = TimeDistributed(Dense(num_classes + 1, name="dense"))(blstm)  # -- 2
    outrnn = Activation('softmax', name='softmax')(dense)

    '''
    Sequence Learning # endregion
    '''

    network = CTCModel([i_vgg, i_keypoint], [outrnn])  # -- 4

    print(network.get_model_train())

    if LOAD_WEIGHT:
        network.load_model(path_dir=f'{MODEL_SAVE_PATH}', file_weights='/model_weights.hdf5', init_last_layer=False,
                           init_archi=False, optimizer=Adam(lr=0.00001))

    # Changed from 0.00001
    # network.compile(optimizer=Adam(lr=0.00001))

    network.summary()

    x_list = []
    y_list = []

    x_len_list = []
    y_len_list = []

    Y_zeros_list = []
    x_key_list = []

    # length from 100 class
    length = 79

    load_npz = np.load(x_data[data_index][0])

    x_skipped = load_npz['arr_0'][::2, :, :, :]

    load_npz = np.pad(x_skipped, ((0, skipped_max_len - x_len[data_index]), (0, 0), (0, 0), (0, 0)), 'constant',
                      constant_values=(0, 0))

    x_npy = np.load(x_data_keypoint[data_index])
    x_npy = x_npy[:, selected, :]

    x_npy_skipped = x_npy[::2, :, :]

    x_npy = x_npy_skipped.reshape((x_npy_skipped.shape[0], 1, 27, 3))
    x_npy = np.pad(x_npy, ((0, skipped_max_len - x_len[data_index]), (0, 0), (0, 0), (0, 0)), 'constant',
                   constant_values=(0, 0))

    label = np.where(y_data[data_index])[1]

    x_key_list.append(np.asarray(x_npy))

    x_list.append(load_npz)

    y_list.append(label)

    # Hardcoded ???
    x_len_list.append(length)

    predict = network.predict_on_batch(
        x=[np.concatenate((np.array(x_list), np.array(x_list)), axis=0),
           np.concatenate((np.array(x_key_list), np.array(x_key_list)), axis=0),
           np.concatenate((np.array(x_len_list), np.array(x_len_list)), axis=0)])

    label_array = load_label()

    sentence_array = list(SENTENCE_LABEL.values())

    result = list(map(lambda x: label_array[int(x)], predict[0]))

    print('=== File Name ===')
    print(file_path)
    print('=== Ground Truth ===')
    print(sentence_array[classes_col[data_index]])
    print('=== Prediction ===')
    print(result)

    return sentence_array[classes_col[data_index]], result, video_col[data_index]

    # wer = calculate_wer(gt=sentence_array[data_index], result=result, length=len(sentence_array[data_index]))
    #
    # print('WER')
    # print(wer)


def predict_ctc_custom(npz_path="F:\\Dataset\\Sign Language\\CSL-Output/000000/P01_s1_00_0._color.npz",
                       npy_path=r"F:\Dataset\Sign Language\CSL-Key\000000\P01_s1_00_0._color.avi.npy", frame_len=106):

    x_data, y_data, x_len, y_len, x_data_keypoint, classes_col, target_data, video_col = generate_data(
        class_count=CLASS_COUNT)

    skipped_max_len = 326
    # 100 total word is 193 words
    num_classes = 193

    print(x_data[0])
    print(y_data[0])
    print(x_len[0])
    print(y_len[0])
    print(x_data_keypoint[0])
    print(classes_col[0])
    print(target_data[0])

    # data_index = np.where(x_data == npz_path)
    data_index = 0
    print('index = ' + str(data_index))
    print(y_data[data_index])
    print(x_len[data_index])
    print(y_len[data_index])
    print(x_data_keypoint[data_index])
    print(classes_col[data_index])
    print(target_data[data_index])
    print(video_col[data_index])

    # if shuffle:
    #     c = list(zip(x_data, y_data))
    #     random.shuffle(c)
    #     x_data, y_data = zip(*c)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(skipped_max_len, 56, 56, 256))

    # Input Keypoint
    i_keypoint = tf.keras.Input(name="input_1_keypoint", shape=(skipped_max_len, 1, 27, 3))
    output_keypoint = TimeDistributed(GlobalAveragePooling2D(name="global_max_full"))(i_keypoint)
    dense_input_keypoint = Dense(256, activation='relu', name='dense_keypoint')(output_keypoint)

    vgg = VGG_2(i_vgg)
    # m_vgg = Model(inputs=[i_vgg], outputs=[vgg])

    # TCN

    '''
    TCN -> Dense
    '''

    o_tcn_full = TCN_layer(vgg, 5)
    o_tcn_keypoint = TCN_layer(dense_input_keypoint, 5)

    # global_pool = GlobalAveragePooling2D(name="global_max_full")(o_tcn_full)
    # flatten = Flatten()(o_tcn_full) # using flatten to sync the network size - disabled if 'TMC FULL'

    dense = Dense(256, name='dense_o_tcn1')(o_tcn_full)
    dense_keypoint = Dense(256, activation='relu', name='dense_o_keypoint')(o_tcn_keypoint)

    '''
    TMC (cont)
    '''

    o_tcn_block1 = TCN_layer(dense, 1)
    o_tcn_block1 = Dense(256, name='dense_o_tcn_intra_block1')(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    block1 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block1)

    o_tcn_key_block1 = TCN_layer(dense_keypoint, 1)
    o_tcn_key_block1 = Dense(256, name='dense_o_tcn_key_intra_block1')(o_tcn_key_block1)
    o_tcn_key_block1 = Dense(256)(o_tcn_key_block1)
    block1_key = MaxPooling1D(pool_size=5, strides=2)(o_tcn_key_block1)

    i_tcn2 = block1
    o_tcn2 = TCN_layer(i_tcn2, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2 = Dense(256, name='dense_o_tcn2')(o_tcn2)
    o_tcn_block2 = TCN_layer(dense2, 1)
    o_tcn_block2 = Dense(512, name='dense_o_tcn_intra_block2')(o_tcn_block2)
    o_tcn_block2 = Dense(512)(o_tcn_block2)
    block2 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block2)

    i_tcn2_key = block1_key
    o_tcn2_key = TCN_layer(i_tcn2_key, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2_key = Dense(256, activation='relu', name='dense_o_tcn2_key')(o_tcn2_key)
    o_tcn_key_block2 = TCN_layer(dense2_key, 1)
    o_tcn_key_block2 = Dense(256, name='dense_o_tcn_key_intra_block2')(o_tcn_key_block2)
    o_tcn_key_block2 = Dense(256)(o_tcn_key_block2)
    block2_key = MaxPooling1D(pool_size=5, strides=2)(o_tcn_key_block2)

    # Concat VGG + Keypoint
    concat = concatenate([block2, block2_key], axis=2)

    '''
    TMC (cont) # endregion

    '''

    '''
    Sequence Learning
    '''

    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(concat)
    dense = TimeDistributed(Dense(num_classes + 1, name="dense"))(blstm)  # -- 2
    outrnn = Activation('softmax', name='softmax')(dense)

    '''
    Sequence Learning # endregion
    '''

    network = CTCModel([i_vgg, i_keypoint], [outrnn])  # -- 4

    print(network.get_model_train())

    if LOAD_WEIGHT:
        network.load_model(path_dir=f'{MODEL_SAVE_PATH}', file_weights='/model_weights.hdf5', init_last_layer=False,
                           init_archi=False, optimizer=Adam(lr=0.00001))

    # Changed from 0.00001
    # network.compile(optimizer=Adam(lr=0.00001))

    network.summary()

    x_list = []
    y_list = []

    x_len_list = []
    y_len_list = []

    Y_zeros_list = []
    x_key_list = []

    # length from 100 class
    length = 79

    # load_npz = np.load(x_data[data_index][0])
    load_npz = np.load(npz_path)

    x_skipped = load_npz['arr_0'][::2, :, :, :]

    load_npz = np.pad(x_skipped, ((0, skipped_max_len - frame_len), (0, 0), (0, 0), (0, 0)), 'constant',
                      constant_values=(0, 0))

    x_npy = np.load(npy_path)
    x_npy = x_npy[:, selected, :]

    x_npy_skipped = x_npy[::2, :, :]

    x_npy = x_npy_skipped.reshape((x_npy_skipped.shape[0], 1, 27, 3))
    x_npy = np.pad(x_npy, ((0, skipped_max_len - frame_len), (0, 0), (0, 0), (0, 0)), 'constant',
                   constant_values=(0, 0))

    label = np.where(y_data[data_index])[1]

    x_key_list.append(np.asarray(x_npy))

    x_list.append(load_npz)

    y_list.append(label)

    # Hardcoded ???
    x_len_list.append(length)

    predict = network.predict_on_batch(
        x=[np.concatenate((np.array(x_list), np.array(x_list)), axis=0),
           np.concatenate((np.array(x_key_list), np.array(x_key_list)), axis=0),
           np.concatenate((np.array(x_len_list), np.array(x_len_list)), axis=0)])

    label_array = load_label()

    sentence_array = list(SENTENCE_LABEL.values())

    result = list(map(lambda x: label_array[int(x)], predict[0]))

    # print('=== File Name ===')
    # print(npz_path)
    # print('=== Ground Truth ===')
    # print(sentence_array[classes_col[data_index]])
    print('=== Prediction ===')
    print(result)

    return sentence_array[classes_col[data_index]], result, video_col[data_index]

    # wer = calculate_wer(gt=sentence_array[data_index], result=result, length=len(sentence_array[data_index]))
    #
    # print('WER')
    # print(wer)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def generate_data(class_count=10, get_frame_length=True):
    sentences = SENTENCE_LABEL

    classes = [i for i in range(class_count)]
    for k in list(sentences):
        if k not in set(classes):
            del sentences[k]
    classes_col = []
    paths_col = []
    path = OUTPUT_PATH
    keypoint_path = KEYPOINT_PATH
    video_path = CSL_PATH

    keypoint_col = []
    video_col = []

    for c in classes:
        paths = os.listdir('{}/{}/'.format(path, str(c).zfill(6)))
        paths = list(map(lambda x: '{}/{}/{}'.format(path, str(c).zfill(6), x), paths))
        keypoint_paths = list(
            map(lambda x: '{}/{}/{}'.format(keypoint_path, str(c).zfill(6), x[43:-4] + '.avi.npy'), paths))
        video_paths = list(
            map(lambda x: '{}/{}/{}'.format(video_path, str(c).zfill(6), x[43:-4] + '.avi'), paths))
        if TESTING:
            keypoint_paths = list(
                map(lambda x: '{}/{}/{}'.format(keypoint_path, str(c).zfill(6), x[48:-4] + '.avi.npy'), paths))
            # keypoint_paths = list(map(lambda x: '{}/{}/{}'.format(keypoint_path, str(c).zfill(6), x[50:-4] + '.avi.npy'), paths))
        if DEBUG:
            print(paths)
            print(keypoint_paths)
            print(video_paths)
            # exit()
        classes_col += [c for i in range(len(paths))]
        paths_col += paths
        keypoint_col += keypoint_paths
        video_col += video_paths

    if DEBUG:
        print(classes_col[0])
        print(paths_col[0])

    input_len = []
    max_vid_len = 0

    print('Calculate Max frame length')
    # for vid in paths_col:
    #     # loaded = np.load(vid)
    #     # print(loaded['arr_0'].shape)
    #     # print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-29:-3] + 'avi')
    #
    #     path = os.path.normpath(vid)
    #     split = path.split(os.sep)
    #
    #     if get_frame_length:
    #         cap = cv2.VideoCapture(
    #             r'F:\Dataset\Sign Language\CSL\pytorch\color/' + split[4] + '/' + split[5][:-3] + 'avi')
    #         # if TESTING:
    #         # cap = cv2.VideoCapture(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-30:-3] + 'avi')
    #         # print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-30:-3] + 'avi')
    #         # exit()
    #         length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     else:
    #         length = 0
    #
    #     input_len.append(math.ceil(length / 2))
    #     if length > max_vid_len:
    #         max_vid_len = length
    #     if DEBUG:
    #         print(length)
    #         print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + split[4] + '/' + split[5][:-3] + 'avi')

    # for vid in paths_col:
    #     # loaded = np.load(vid)
    #     # print(loaded['arr_0'].shape)
    #     # print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-29:-3] + 'avi')
    #
    #     path = os.path.normpath(vid)
    #     split = path.split(os.sep)
    #
    #     if get_frame_length:
    #         cap = cv2.VideoCapture(
    #             r'F:\Dataset\Sign Language\CSL\pytorch\color/' + split[4] + '/' + split[5][:-3] + 'avi')
    #         # if TESTING:
    #         # cap = cv2.VideoCapture(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-30:-3] + 'avi')
    #         # print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-30:-3] + 'avi')
    #         # exit()
    #         length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     else:
    #         length = 0
    #
    #     input_len.append(math.ceil(length / 2))
    #     if length > max_vid_len:
    #         max_vid_len = length
    #     if DEBUG:
    #         print(length)
    #         print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + split[4] + '/' + split[5][:-3] + 'avi')

    frame_len = np.load(r"F:\Dataset\Sign Language\Demo CSL\frame_len.npy")
    print(frame_len[:2500])
    # Trim 10 class
    # lengths = frame_len
    lengths = frame_len[:2500]

    for vid in lengths:
        if get_frame_length:
            length = int(vid)
        else:
            length = 0

        input_len.append(length)

    print('Calculate Max frame length done.')

    print(input_len)
    print(max_vid_len)
    skipped_max_vid_len = math.ceil(max_vid_len / 2)
    print(skipped_max_vid_len)

    # exit()

    # y_data = tf.keras.utils.to_categorical(classes_col, num_classes=5)

    # print(y_data[0])

    y_df, x_df = pd.DataFrame(classes_col), pd.DataFrame(paths_col)

    y_df = y_df.apply(lambda x: sentences[x[0]], axis=1)
    x_df.shape, y_df.shape

    # print(x_df[0])
    # print(y_df[0])
    # print(classes_col)
    # print(paths_col)

    from sklearn.preprocessing import LabelEncoder
    import functools
    target_tokens = list(functools.reduce(lambda a, b: a.union(b), list(map(lambda x: set(x[1]), sentences.items()))))
    # target_tokens += ['#START', '#END']
    num_classes = len(target_tokens)
    sentences_array = list(map(lambda x: x[1], sentences.items()))
    max_sentence = len(max(sentences_array, key=lambda x: len(x)))
    max_sentence = len(max(sentences_array, key=lambda x: len(x))) + 2
    le = LabelEncoder()
    le.fit(target_tokens)

    print(num_classes)

    # sentences_y = y_df.apply(lambda x: ['#START'] + x + ['#END'])
    sentences_y = y_df.apply(lambda x: x)
    print(y_df)
    print(sentences_y)

    sentences_y_encoded = sentences_y.apply(lambda x: le.transform(x))
    decoder_input_data = np.zeros((len(sentences_y_encoded),
                                   max_sentence, num_classes))
    decoder_target_data = np.zeros((len(sentences_y_encoded),
                                    max_sentence, num_classes))
    for i in range(decoder_input_data.shape[0]):
        for t in range(sentences_y_encoded[i].shape[0]):
            decoder_input_data[i, t, sentences_y_encoded[i][t]] = 1
            if t > 0:
                decoder_target_data[i, t - 1, sentences_y_encoded[i][t]] = 1
    decoder_target_data.shape, decoder_input_data.shape

    # build dict of indexes
    classes = le.classes_
    transformed = le.transform(classes)
    decoder_indexes = dict((classes[i], transformed[i]) for i in range(len(classes)))
    reverse_decoder_indexes = dict((transformed[i], classes[i]) for i in range(len(classes)))

    if DEBUG:
        print(decoder_input_data[0])
        print(decoder_target_data[0])
        print(sentences_y[0])

    # print(decoder_indexes)
    # print('reverse indexes')
    # print(reverse_decoder_indexes)

    label_len = []
    # input_len = []

    for labels in decoder_input_data:
        count = 0
        for label in labels:
            # print(label)
            if 1 in label:
                count += 1
        # print(count)
        label_len.append(count)
        # input_len.append(count)

    from sklearn.model_selection import train_test_split
    sentences_x_train, sentences_x_validation, decoder_input_train, \
    decoder_input_val, decoder_target_train, decoder_target_val, \
    sentences_y_train, sentences_y_validation, label_len_train, label_len_val, input_len_train, input_len_val, \
    keypoint_x_train, keypoint_x_validation \
        = train_test_split(x_df, decoder_input_data, decoder_target_data, sentences_y, label_len, input_len,
                           keypoint_col,
                           test_size=0.2)
    print('Decoder Train')
    print(sentences_x_train)
    print(decoder_input_train)
    print(label_len_train)

    print(sentences_y_train)
    # print(decoder_target_train[0])

    # return np.asarray(sentences_x_train), decoder_input_train, input_len_train, label_len_train, \
    #        np.asarray(
    #            sentences_x_validation), decoder_input_val, input_len_val, label_len_val, keypoint_x_train, keypoint_x_validation, \
    #        max_vid_len, num_classes, skipped_max_vid_len

    return np.asarray(
        x_df), decoder_input_data, input_len, label_len, keypoint_col, classes_col, decoder_target_data, video_col


def verify_npz():
    folders = [f.path for f in os.scandir(OUTPUT_PATH) if f.is_dir()]

    print(folders)

    for sentence in range(SENTENCE_START, SENTENCE_END):
        # print(f'{CSL_PATH}\{str(sentence).zfill(6)}')
        print(folders[sentence])

        files = [f for f in listdir(folders[sentence]) if isfile(join(folders[sentence], f))]

        for file in files:
            filename = fr'{folders[sentence]}\{file}'
            print(filename)
            x_npz = np.load(filename, mmap_mode='r')

            print(x_npz['arr_0'])

            gc.collect()


from sklearn.preprocessing import LabelEncoder
import functools


def load_label():
    label = load_obj('reverse_decoder_indexes_100class')

    print(list(label.values()))

    return list(label.values())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load_data() # crop + output 4th layer
    # generate_data()
    # verify_npz()
    # testing()
    # label_generator()
    # load_label()

    # predict_ctc(file_path="F:\\Dataset\\Sign Language\\CSL-Output/000005/P01_s1_05_3._color.npz")
    # predict_ctc()
    predict_ctc_custom()

    # predict_ctc_custom()

    # load_label()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
