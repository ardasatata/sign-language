import errno
import gc
import glob
import math
import os
import sys
from os import listdir, path
from os.path import isfile, join, isdir

from tensorflow.keras.utils import Progbar
from keras_ctcmodel.CTCModel import CTCModel as CTCModel

import cv2
import csv

import re

import imageio

# import progressbar
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
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.backend import concatenate

from LSA64 import calculate_wer
from extract_layer4 import get_output_layer

import numpy as np
from numpy import savez_compressed

import pandas as pd
import os

from tqdm import tqdm

CSL_PATH = r'D:\Dataset\Sign Language\CSL\pytorch\color'

# OUTPUT_PATH = r'F:\Dataset\Sign Language\CSL-Output'
# OUTPUT_PATH = r'F:\Dataset\Sign Language\CSL-Output'
# OUTPUT_PATH = r'F:\Dataset\Sign Language\CSL-Output-ResNet'
OUTPUT_PATH = r'D:\Dataset\Sign Language\CSL-Output_test'
KEYPOINT_PATH = r'D:\Dataset\Sign Language\CSL-Key_test'
MODEL_SAVE_PATH = r"D:\Dataset\Sign Language\CSL Model + Keypoint - Test"
# MODEL_SAVE_PATH = r"F:\Dataset\Sign Language\CSL Model + Keypoint Resnet"
CLASS_COUNT = 10

SENTENCE_START = 70
SENTENCE_END = 100
SAMPLE_PER_SENTENCE = 250

PREVIEW = False
DEBUG = False
TESTING = False

LOAD_WEIGHT = False

# # TESTING #
# KEYPOINT_PATH = r'F:\Dataset\Sign Language\CSL-Key_test'
# OUTPUT_PATH = r'F:\Dataset\Sign Language\CSL-Output_test'
# MODEL_SAVE_PATH = r"F:\Dataset\Sign Language\CSL Model + Keypoint - Test"
# CLASS_COUNT = 2
# TESTING = True
# # DEBUG = True

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

    # out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x, x)
    # x = Flatten()(x)

    return x


def train_ctc(shuffle=True):
    x_data, y_data, x_len, y_len, x_data_val, y_data_val, x_len_val, y_len_val, x_data_keypoint, x_data_keypoint_validate, max_len, num_classes, skipped_max_len = generate_data(
        class_count=CLASS_COUNT)

    # if shuffle:
    #     c = list(zip(x_data, y_data))
    #     random.shuffle(c)
    #     x_data, y_data = zip(*c)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(100, 56, 56, 256))

    # Input Keypoint
    i_keypoint = tf.keras.Input(name="input_1_keypoint", shape=(100, 1, 27, 3))
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

    block2_attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=4, name="block2_attn")(block2, block2)

    i_tcn2_key = block1_key
    o_tcn2_key = TCN_layer(i_tcn2_key, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2_key = Dense(256, activation='relu', name='dense_o_tcn2_key')(o_tcn2_key)
    o_tcn_key_block2 = TCN_layer(dense2_key, 1)
    o_tcn_key_block2 = Dense(256, name='dense_o_tcn_key_intra_block2')(o_tcn_key_block2)
    o_tcn_key_block2 = Dense(256)(o_tcn_key_block2)
    block2_key = MaxPooling1D(pool_size=5, strides=2)(o_tcn_key_block2)

    # block2_key_attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=4, name="block2_key_attn")(block2_key, block2_key)

    # Concat VGG + Keypoint
    # concat = concatenate([block2_attn, block2_key_attn], axis=2)

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
        network.load_model(path_dir=f'{MODEL_SAVE_PATH}', file_weights='/model_weights.hdf5',
                           optimizer=Adam(0.00001), init_last_layer=False, init_archi=False)

    network.compile(optimizer=Adam(lr=0.00001))

    network.summary()

    metrics_names = ['val']

    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)

    batch_size = 2
    epochs = 60

    loss_ = 999999999

    acc_model = []
    loss_model = []
    error_model = []

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch + 1}')

        acc = []
        loss = []
        wers = []

        pb_i = Progbar(len(x_data), stateful_metrics=metrics_names)

        y_zeros = np.zeros(len(x_data))
        y_zeros_validate = np.zeros(len(x_data_keypoint_validate))

        for i in range(0, len(x_data) // batch_size):
            X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
            y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]

            X_len = x_len[i * batch_size:min(len(x_len), (i + 1) * batch_size)]
            Y_len = y_len[i * batch_size:min(len(y_len), (i + 1) * batch_size)]

            Y_zeros = y_zeros[i * batch_size:min(len(y_zeros), (i + 1) * batch_size)]
            X_keypoint = x_data_keypoint[i * batch_size:min(len(x_data_keypoint), (i + 1) * batch_size)]

            x_list = []
            y_list = []

            x_len_list = []
            y_len_list = []

            Y_zeros_list = []
            x_key_list = []

            # TODO Get value from network softmax Layer
            length = 28

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                # x_seq, y_seq = sentence_sequence_generator_npz(X[i_data], y[i_data])


                print(np.where(y[i_data])[1])
                print(length)
                print(np.asarray(Y_len[i_data]))

                exit()

                load_npz = np.load(X[i_data][0])

                if DEBUG:
                    print(X[i_data][0])
                    print(X_keypoint[i_data])
                    print(X_len[i_data])

                # print(np.array(load_npz['arr_0']).shape)
                # print(np.array(load_npz['arr_0'][::2, :, :, :]).shape)

                x_skipped = load_npz['arr_0'][::2, :, :, :]

                load_npz = np.pad(x_skipped, ((0, skipped_max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]

                x_npy_skipped = x_npy[::2, :, :]

                x_npy = x_npy_skipped.reshape((x_npy_skipped.shape[0], 1, 27, 3))
                x_npy = np.pad(x_npy, ((0, skipped_max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))
                x_key_list.append(np.asarray(x_npy))

                x_list.append(load_npz)
                # y_list.append(np.asarray(y[i_data]))

                # x_len_list.append(np.asarray(X_len[i_data]))

                label = np.where(y[i_data])[1]

                y_list.append(label)

                # Hardcoded ???
                x_len_list.append(length)
                y_len_list.append(np.asarray(Y_len[i_data]))

                Y_zeros_list.append(Y_zeros[i_data])

                if DEBUG:
                    print(x_npy.shape)
                    print(load_npz.shape)

            # label = np.where(y_list)[2]
            #
            # y_test = []
            #
            # y_test.append(label)

            # print(label)

            y_list = tf.keras.preprocessing.sequence.pad_sequences(
                y_list, padding="post"
            )

            input = [np.array(x_list), np.array(x_key_list), np.array(y_list), np.array(x_len_list),
                     np.array(y_len_list)]

            history = network.train_on_batch(
                x=input,
                y=np.array(Y_zeros_list))

            # history = network.fit(
            #     x=[np.array(x_list), np.array(y_list), np.array(x_len_list), np.array(y_len_list)], y=np.array(Y_zeros_list), batch_size=1)

            # print(f'history : {history}')

            values = [('val', history)]

            pb_i.add(batch_size, values=values)

            # acc.append(history[1])
            # loss.append(history[0])

            # print(f'Loss : {X[0]}')
            # print(X)

        pb_val = Progbar(len(x_data_keypoint_validate), stateful_metrics=['wer'])

        # Validate dataset
        # for i in range(0, len(x_data_keypoint_validate) // batch_size):
        for i in range(0, len(x_data_keypoint_validate) // 1):
            X = x_data_val[i * 1:min(len(x_data_val), (i + 1) * 1)]
            y = y_data_val[i * 1:min(len(y_data_val), (i + 1) * 1)]

            X_len = x_len_val[i * 1:min(len(x_len_val), (i + 1) * 1)]
            Y_len = y_len_val[i * 1:min(len(y_len_val), (i + 1) * 1)]

            Y_zeros = y_zeros_validate[i * 1:min(len(y_zeros_validate), (i + 1) * 1)]
            X_keypoint = x_data_keypoint_validate[
                         i * 1:min(len(x_data_keypoint_validate), (i + 1) * 1)]

            x_list = []
            y_list = []

            x_len_list = []
            y_len_list = []

            Y_zeros_list = []
            x_key_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                # x_seq, y_seq = sentence_sequence_generator_npz(X[i_data], y[i_data])

                load_npz = np.load(X[i_data][0])

                x_skipped = load_npz['arr_0'][::2, :, :, :]

                load_npz = np.pad(x_skipped, ((0, skipped_max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]

                x_npy_skipped = x_npy[::2, :, :]

                x_npy = x_npy_skipped.reshape((x_npy_skipped.shape[0], 1, 27, 3))
                x_npy = np.pad(x_npy, ((0, skipped_max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))
                x_key_list.append(np.asarray(x_npy))

                x_list.append(load_npz)
                # y_list.append(np.asarray(y[i_data]))

                # x_len_list.append(np.asarray(X_len[i_data]))

                label = np.where(y[i_data])[1]

                y_list.append(label)

                # Hardcoded ???
                x_len_list.append(length)
                y_len_list.append(np.asarray(Y_len[i_data]))

                Y_zeros_list.append(Y_zeros[i_data])

            # label = np.where(y_list)[2]
            #
            # y_test = []
            #
            # y_test.append(label)

            # print(label)

            # y_list = tf.keras.preprocessing.sequence.pad_sequences(
            #     y_list, padding="post"
            # )
            # print(np.array(x_list).shape)
            # print(np.array(x_key_list).shape)
            # print(np.array(x_len_list).shape)

            # x_list = np.array(x_list).reshape((1, np.array(x_list).shape[0],  np.array(x_list).shape[1],  np.array(x_list).shape[2],  np.array(x_list).shape[3]))
            # x_key_list = np.array(x_key_list).reshape((1, np.array(x_key_list).shape[0],  np.array().shape[1],  np.array(x_key_list).shape[2],  np.array(x_key_list).shape[3]))
            # x_len_list = np.array(x_len_list).reshape((1, np.array(x_len_list).shape[0]))

            predict = network.predict_on_batch(
                x=[np.concatenate((np.array(x_list), np.array(x_list)), axis=0), np.concatenate((np.array(x_key_list), np.array(x_key_list)), axis=0),
                   np.concatenate((np.array(x_len_list), np.array(x_len_list)), axis=0)])
            y_new = np.concatenate((np.array(y_list), np.array(y_list)), axis=0)

            # predict = network.predict_on_batch(x=[np.array(x_list), np.array(x_key_list), np.array(x_len_list)])
            # y_new = np.array(y_list ,dtype=object)

            # print(predict[0])
            # print(y_new[0])
            #
            # print(predict)
            # print(y_new)
            #
            # print(len(y_list[0]))

            if DEBUG:
                print(y_new)
                print(predict)
                print(len(y_list[0]))

            wer = calculate_wer(gt=y_new, result=predict, length=len(y_list[0]))
            wers.append(wer)

            values = [('wer', wer)]
            pb_val.add(1, values=values)

        wers_avg = np.average(wers)
        error_model.append(wers_avg)
        print('#########')
        print(f'WER AVG / EPOCH {epoch + 1} : {wers_avg}')
        print('#########')

        network.save_model(f'{MODEL_SAVE_PATH}/')

        # loaded.load_model(r'F:\Dataset\Sign Language\CSL Model + Keypoint\\', optimizer=Adam(lr=0.00001), init_archi=False)
        # loaded.summary()

        # print(f'Loss : {np.average(np.array(loss))}, Accuracy : {np.average(np.array(acc))}')
        # acc_model.append(np.average(np.array(acc)))
        # loss_model.append(np.average(np.array(loss)))

        # if epoch > 1:
        #     xd, yd, xl, yl = sentence_dataset_generator(sample_number=1)
        #
        #     test_data = []
        #
        #     for i in range(0, len(SENTENCE_ARRAY)):
        #         xs, ys = sentence_sequence_generator_npz(xd[i], yd[i])
        #         test_data.append(xs)
        #
        #     # print(np.array(test_data).shape)
        #
        #     result = network.predict(x=[np.array(test_data), np.array([5, 5, 5, 5, 5])])
        #     words = np.where(result != -1)
        #     print("word", words)
        #     print(result)
        #     print(np.asarray(result[0]).shape)

        # if np.average(np.array(loss)) < loss_:
        #     loss_ = np.average(np.array(loss))
        #     network.save(
        #         filepath=f'{MODEL_SAVE_PATH}{"CTC_FULL_LSA64"}_{loss_}.h5')

        # val_loss = []
        # for i in range(0, len(x_data) // batch_size):
        #     X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
        #     y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]
        #     val_loss.append(tcn_model.validate_on_batch(X, y))
        #
        # print('Validation Loss: ' + str(np.mean(val_loss)))

    # network.save_model(path_dir=f'{MODEL_SAVE_PATH}')

    print(acc_model)
    print(loss_model)

    error_avg = np.average(error_model)
    print('#########')
    print(f'model WER : {error_avg}')
    print('######### xoxo #########')
    print('train done')


def generate_data(class_count=10):
    sentences = {0: ['他的', '同学', '是', '警察'], 1: ['他', '妈妈', '的', '同学', '是', '公务', '员'], 2: ['我的', '爸爸', '是', '商人'],
                 3: ['他', '哥哥', '的', '目标', '是', '解放', '军'], 4: ['他', '姐姐', '的', '目标', '是', '模特'],
                 5: ['我', '朋友', '的', '祖父', '是', '工人'], 6: ['我', '同学', '的', '妈妈', '是', '保姆'],
                 7: ['我', '同学', '的', '妹妹', '是', '护士'], 8: ['我的', '妻子', '是', '教师'], 9: ['我的', '丈夫', '是', '导演'],
                 10: ['他', '哥哥', '的', '朋友', '是', '演员'], 11: ['他的', '女朋友', '是', '护士'],
                 12: ['她', '丈夫', '的', '朋友', '是', '教练'], 13: ['他', '哥哥', '的', '同学', '是', '医生'],
                 14: ['他', '妹妹', '的', '同学', '是', '律师'], 15: ['他的', '邻居', '是', '残疾', '人'], 16: ['他的', '外祖母', '是', '园丁'],
                 17: ['他的', '祖父', '是', '炊事员'], 18: ['我', '表哥', '的', '邻居', '是', '记者'], 19: ['他的', '丈夫', '是', '警察'],
                 20: ['我', '公公', '是', '牧民'], 21: ['我', '婆婆', '是', '保育', '员'], 22: ['他', '儿子', '是', '弱', '智', '人'],
                 23: ['我', '嫂嫂', '是', '画家'], 24: ['你', '祖父', '是', '知识', '分子'], 25: ['你', '外祖父', '是', '猎', '手'],
                 26: ['他', '爸爸', '是', '保安'], 27: ['他', '妈妈', '是', '裁缝'], 28: ['他', '外祖父', '是', '邮递', '员'],
                 29: ['他', '祖母', '是', '聋人'], 30: ['我', '公公', '是', '门', '卫'], 31: ['你', '妹妹', '是', '会计'],
                 32: ['你', '哥哥', '是', '武', '警'], 33: ['你', '婆婆', '是', '盲人'], 34: ['你', '爸爸', '是', '编辑'],
                 35: ['她', '外祖父', '是', '农民'], 36: ['我', '儿子', '是', '职员'], 37: ['他', '弟弟', '是', '向导'],
                 38: ['他', '岳父', '是', '残疾', '人'], 39: ['他', '姐夫', '是', '刑警'], 40: ['我的', '新', '尺子', '是', '歪', '的'],
                 41: ['提高', '他的', '社会', '地位'], 42: ['紧张', '的', '工作', '气氛'], 43: ['工作', '效益', '稳定', '提高'],
                 44: ['工作', '环境', '的', '改善'], 45: ['民主', '团结', '的', '局势'], 46: ['我的', '毛毯', '是', '新的'],
                 47: ['现实', '情况', '我', '容易', '紧张'], 48: ['形成', '稳定', '的', '基础'], 49: ['社会', '团结', '是', '基础'],
                 50: ['我们', '的', '国家', '富强', '民主'], 51: ['国民', '的', '婚姻', '幸福'], 52: ['我们', '是', '自由', '恋爱'],
                 53: ['我们', '捐献', '的是', '毛毯'], 54: ['茶壶', '是', '褐色', '的'], 55: ['他的', '工作', '是', '美容'],
                 56: ['搞清', '形势', '是', '困难', '的'], 57: ['发挥', '我们', '的', '优势'], 58: ['扭转', '局面', '是', '困难', '的'],
                 59: ['现实', '是', '学生', '任务', '多'], 60: ['社会', '的', '安定'], 61: ['地球', '是', '行星'],
                 62: ['月亮', '是', '地球', '的', '卫星'], 63: ['太阳', '是', '恒星'], 64: ['国家', '经济', '情况', '好'],
                 65: ['他的', '前途', '事业', '成功'], 66: ['他们', '的', '国家', '摆脱', '贫苦'], 67: ['社会', '地位', '的', '提高'],
                 68: ['颜色', '是', '丰富', '的'], 69: ['社会', '就业', '形势', '好'], 70: ['他的', '工作', '经验', '丰富'],
                 71: ['剪刀', '是', '锋利', '的'], 72: ['新的', '被子', '是', '破', '的'], 73: ['天气', '预报', '有', '雨'],
                 74: ['他的', '盆', '是', '绿', '的'], 75: ['杯子', '是', '橙色', '的'], 76: ['加强', '国家', '的', '保卫'],
                 77: ['他', '放弃', '目标'], 78: ['引导', '他人', '成功'], 79: ['结果', '圆满', '成功'], 80: ['他的', '手表', '是', '坏', '的'],
                 81: ['他的', '牙刷', '疏'], 82: ['洗脸盆', '是', '空', '的'], 83: ['招聘', '的', '岗位', '多'], 84: ['我们', '拜访', '外祖父'],
                 85: ['事情', '有', '改善'], 86: ['我们', '的', '友谊', '深'], 87: ['裁缝', '有', '针线'], 88: ['妈妈', '有', '项链'],
                 89: ['我有', '打火', '机'], 90: ['天空', '没有', '星星'], 91: ['他的', '小孩子', '有', '礼貌'], 92: ['他', '招呼', '你来'],
                 93: ['观察', '他的', '情况'], 94: ['我', '推荐', '他', '去', '就业'], 95: ['手电筒', '有', '电池'],
                 96: ['我们', '的', '婚姻', '是', '幸福', '的'], 97: ['国家', '稳定', '是', '幸福', '的', '基础'],
                 98: ['社会', '地位', '是', '平等', '的'], 99: ['我的', '毛巾', '是', '干', '的']}

    classes = [i for i in range(class_count)]
    for k in list(sentences):
        if k not in set(classes):
            del sentences[k]
    classes_col = []
    paths_col = []
    path = OUTPUT_PATH
    keypoint_path = KEYPOINT_PATH
    keypoint_col = []

    for c in classes:
        paths = os.listdir('{}/{}/'.format(path, str(c).zfill(6)))
        paths = list(map(lambda x: '{}/{}/{}'.format(path, str(c).zfill(6), x), paths))
        keypoint_paths = list(
            map(lambda x: '{}/{}/{}'.format(keypoint_path, str(c).zfill(6), x[70:-4] + '.avi.npy'), paths))
        if TESTING:
            keypoint_paths = list(
                map(lambda x: '{}/{}/{}'.format(keypoint_path, str(c).zfill(6), x[68:-4] + '.avi.npy'), paths))
            # keypoint_paths = list(map(lambda x: '{}/{}/{}'.format(keypoint_path, str(c).zfill(6), x[50:-4] + '.avi.npy'), paths))
        if DEBUG:
            print(paths)
            print(keypoint_paths)
            # exit()
        classes_col += [c for i in range(len(paths))]
        paths_col += paths
        keypoint_col += keypoint_paths

    if DEBUG:
        print(classes_col[0])
        print(paths_col[0])

    input_len = []
    max_vid_len = 0

    print('Calculate Max frame length')
    for vid in paths_col:
        # loaded = np.load(vid)
        # print(loaded['arr_0'].shape)
        # print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-29:-3] + 'avi')

        path = os.path.normpath(vid)
        split = path.split(os.sep)

        cap = cv2.VideoCapture(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + split[4] + '/' + split[5][:-3] + 'avi')
        # if TESTING:
        # cap = cv2.VideoCapture(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-30:-3] + 'avi')
        # print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-30:-3] + 'avi')
        # exit()
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_len.append(math.ceil(length / 2))
        if length > max_vid_len:
            max_vid_len = length
        if DEBUG:
            print(length)
            print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + split[4] + '/' + split[5][:-3] + 'avi')
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

    len(target_tokens)
    print(num_classes)

    # sentences_y = y_df.apply(lambda x: ['#START'] + x + ['#END'])
    sentences_y = y_df.apply(lambda x: x)
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

    print(decoder_indexes)
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
                           test_size=0.2, random_state=1)
    print('Decoder Train')
    print(sentences_x_train)
    print(decoder_input_train)
    print(label_len_train)

    print(sentences_y_train)
    # print(decoder_target_train[0])

    return np.asarray(sentences_x_train), decoder_input_train, input_len_train, label_len_train, \
           np.asarray(
               sentences_x_validation), decoder_input_val, input_len_val, label_len_val, keypoint_x_train, keypoint_x_validation, \
           max_vid_len, num_classes, skipped_max_vid_len


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


def testing():
    print('testing')
    # F:\Dataset\Sign Language\CSL\pytorch\color/00090/P01_s10_00_0._color.avi

    cap = cv2.VideoCapture(r'F:\Dataset\Sign Language\CSL\pytorch\color/000099/P01_s10_00_0._color.avi')

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)

    filename = 'F:\Dataset\Sign Language\CSL\pytorch\color/000099/P01_s10_00_0._color.avi'
    vid = imageio.get_reader(filename, 'ffmpeg')
    # number of frames in video
    num_frames = vid._meta['nframes']

    print(num_frames)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load_data() # crop + output 4th layer
    # generate_data()
    train_ctc()
    # verify_npz()
    # testing()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
