import glob
import math
import os
from os import listdir
from os.path import isfile, join
from jiwer import wer

import gc

# import livelossplot
import livelossplot as livelossplot
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
from train_custom import VGG, VGG_2

from tensorflow.python.keras.backend import concatenate

from keras.models import load_model

from keras.optimizers import Adam, SGD

from keras_ctcmodel.CTCModel import CTCModel as CTCModel

from keras import backend as K

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
from matplotlib import pyplot

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

MODEL = r"D:\TSL\.h5"

MODEL_SAVE_PATH = r'F:\Dataset\Sign Language\LSA64_MODEL\\'
CTC_MODEL_PATH = r'C:\Users\minelab\dev\TSL\model\LSA64\CTC\\'

DIR = r'F:\Arda\LSA64\LSA64_Cropped'

# DIR = r'E:\arda\LSA_64'

SAVE_PATH = r'D:\LSA64\LSA64_Cropped\\'
NPZ_DIR = r'D:\LSA64\VGG_out_new'

ORIGINAL_VIDEO_PATH = r'F:\Arda\LSA64\LSA64_Cropped\\'

# 10 Class with full sample & subject

# # Testing
# TOTAL_CLASS = 2
# TOTAL_SUBJECT = 2
# TOTAL_SAMPLE = 2

RESOLUTION = 224

FILENAME_PADDING = 15

EXT = r'mp4'

MAX_FRAME = 18

EXTRACT_DESTINATION = r'D:\LSA64\VGG_out_new\\'
EXTRACT_DESTINATION = r'F:\Arda\LSA64\OUTPUT_VGG\\'

# TRAINING CONFIG

EPOCH = 25
BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 1

# Class Range 0 to 64
START_FROM = 0
TOTAL_CLASS = 64

# MAX 10
TOTAL_SUBJECT = 10
# MAX 5
TOTAL_SAMPLE = 5

PREVIEW = False
DEBUG = False
TESTING = False
LOAD_MODEL = False

OUTPUT_PATH = r'F:\Arda\LSA64\OUTPUT_NEW'
KEYPOINT_PATH = r'F:\Arda\LSA64\LSA64_Key'

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


def generate_data_list():
    print("Generate Dataset")

    x_data = []
    x_npz = []
    x_keypoint = []
    y_data = []

    for class_num in range(START_FROM, TOTAL_CLASS):
        for subject_num in range(0, TOTAL_SUBJECT):
            for sample_num in range(0, TOTAL_SAMPLE):
                file_npz = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                    .format(dir=OUTPUT_PATH, class_num=str(class_num + 1).zfill(3),
                            subject_num=str(subject_num + 1).zfill(3),
                            sample_num=str(sample_num + 1).zfill(3), ext='npz')
                file = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                    .format(dir=DIR, class_num=str(class_num + 1).zfill(3), subject_num=str(subject_num + 1).zfill(3),
                            sample_num=str(sample_num + 1).zfill(3), ext='avi')
                file_keypoint = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                    .format(dir=KEYPOINT_PATH, class_num=str(class_num + 1).zfill(3),
                            subject_num=str(subject_num + 1).zfill(3),
                            sample_num=str(sample_num + 1).zfill(3), ext='avi.npy')

                if DEBUG:
                    print(file)
                    print(file_npz)
                    print(file_keypoint)

                x_data.append(file)
                x_npz.append(file_npz)
                x_keypoint.append(file_keypoint)
                y_data.append(class_num)

    input_len = []
    max_vid_len = 0

    print('Calculate Max frame length')
    for vid in x_data:

        cap = cv2.VideoCapture(vid)
        # if TESTING:
        # cap = cv2.VideoCapture(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-30:-3] + 'avi')
        # print(r'F:\Dataset\Sign Language\CSL\pytorch\color/' + vid[-30:-3] + 'avi')
        # exit()
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_len.append(length)
        if length > max_vid_len:
            max_vid_len = length
        if DEBUG:
            print(length)
            print(vid)
    print('Calculate Max frame length done.')

    print(x_npz)
    print(x_keypoint)
    print(x_data)
    print(input_len)
    print(y_data)
    print(max_vid_len)

    from sklearn.model_selection import train_test_split

    y_data = tf.keras.utils.to_categorical(y_data, num_classes=TOTAL_CLASS)

    x_npz_train, x_npz_val, y_data_train, y_data_validation, x_keypoint_train, x_keypoint_val, x_len_train, x_len_val = train_test_split(
        x_npz,
        y_data,
        x_keypoint,
        input_len,
        test_size=0.2, random_state=2)

    print(x_npz_train)
    print(x_npz_val)
    print(y_data_train)
    print(y_data_validation)

    return x_npz_train, x_npz_val, y_data_train, y_data_validation, x_keypoint_train, x_keypoint_val, x_len_train, x_len_val, max_vid_len


from tqdm import tqdm


def convert_data():
    files = [f for f in listdir(ORIGINAL_VIDEO_PATH) if isfile(join(ORIGINAL_VIDEO_PATH, f))]

    print(files)

    with tqdm(total=3200) as bar:
        for idx, file in enumerate(files):

            filename = ORIGINAL_VIDEO_PATH + file

            print(filename)

            if DEBUG:
                print(file)
                print(filename)

            save_file = f'{OUTPUT_PATH}\{file[:-4]}.npz'

            if isfile(save_file):
                print('exist ' + save_file)
                continue

            try:
                cap = cv2.VideoCapture(filename)
                video = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    # (height, width) = frame.shape[:2]
                    if ret:

                        if PREVIEW:
                            cv2.imshow('orig', frame)

                        # cropped = frame[0 + CROP_TOP:720, 0 + CROP_X:1280 - CROP_X]
                        #
                        # if PREVIEW:
                        #     cv2.imshow('cropped', cropped)

                        resized_image = cv2.resize(frame, (224, 224))

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

                # save_dir = f'{OUTPUT_PATH}\{folderName}'

                # if isdir(save_dir):
                #     print(save_dir)
                #     print('exist')
                #     exit(0)
                #     savez_compressed(f'{OUTPUT_PATH}\{folderName}\{fileName}.npz', output)
                # # else :

                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)

                savez_compressed(save_file, output)

                # exit(0)

                cap.release()

            except cv2.error as e:
                print(e)
                False

            tf.keras.backend.clear_session()

            # for idx, file in enumerate(files):
            #     crop_video(f'{folders[sentence]}\{file}', file[:-4], str(sentence).zfill(6))
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


skipped_max_len = 120


def train_multi():
    x_data, x_data_val, y_data, y_data_val, x_data_keypoint, x_data_keypoint_validate, x_len, x_len_val, max_len = generate_data_list()

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(max_len, 56, 56, 256))

    # Input Keypoint
    i_keypoint = tf.keras.Input(name="input_1_keypoint", shape=(max_len, 1, 27, 3))
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

    flatten = Flatten()(concat)  # using flatten to sync the network size - disabled if 'TMC FULL'

    # Added 2 dense layer

    dense_concat = Dense(4096)(flatten)

    dense_concat = Dense(256)(dense_concat)

    dense = Dense(TOTAL_CLASS, activation='softmax')(dense_concat)

    network = Model(inputs=[i_vgg, i_keypoint], outputs=dense)

    if LOAD_MODEL:
        print('Load previous training')
        network = tf.keras.models.load_model(f'{MODEL_SAVE_PATH}LSA64_acc_0.609375_epoch_20.h5')

    network.summary()

    metrics = ['accuracy']

    network.compile(optimizer=Adam(lr=0.0001,
                                   beta_1=0.96,
                                   beta_2=0.999
                                   ), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

    if DEBUG:
        print(np.asarray(x_data).shape)
        print(np.asarray(x_data_keypoint).shape)
        print(np.asarray(y_data).shape)

    batch_size = BATCH_SIZE
    validation_batch_size = VALIDATION_BATCH_SIZE
    epochs = EPOCH

    loss_ = 999999999

    acc_model = []
    loss_model = []
    error_model = []

    metrics_names = ['loss', 'accuracy']

    acc_ = 0

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch + 1}')

        acc = []
        loss = []
        wers = []

        pb_i = Progbar(len(x_data), stateful_metrics=metrics_names)

        for i in range(0, len(x_data) // batch_size):
            X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
            y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]

            X_keypoint = x_data_keypoint[i * batch_size:min(len(x_data_keypoint), (i + 1) * batch_size)]

            X_len = x_len[i * batch_size:min(len(x_len), (i + 1) * batch_size)]

            x_list = []
            y_list = []
            x_len_list = []
            x_key_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                # x_seq, y_seq = sentence_sequence_generator_npz(X[i_data], y[i_data])

                load_npz = np.load(X[i_data])

                if DEBUG:
                    print(X[i_data][0])
                    print(X_keypoint[i_data])

                x_npz = load_npz['arr_0']

                load_npz = np.pad(x_npz, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]

                x_npy = x_npy.reshape((x_npy.shape[0], 1, 27, 3))
                x_npy = np.pad(x_npy, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))

                x_key_list.append(np.asarray(x_npy))
                x_list.append(load_npz)

                y_list.append(y[i_data])

                if DEBUG:
                    print(x_npy.shape)
                    print(load_npz.shape)

            if DEBUG:
                print(np.array(x_list).shape)
                print(np.array(x_key_list).shape)

            history = network.train_on_batch(x=[np.array(x_list), np.array(x_key_list)], y=np.array(y_list))

            # history = network.fit(
            #     x=[np.array(x_list), np.array(y_list), np.array(x_len_list), np.array(y_len_list)], y=np.array(Y_zeros_list), batch_size=1)

            # print(f'history : {history}')

            values = [('loss', history[0]),('accuracy', history[1])]

            pb_i.add(batch_size, values=values)

            # acc.append(history[1])
            # loss.append(history[0])

            # print(f'Loss : {X[0]}')
            # print(X)

        pb_val = Progbar(len(x_data_keypoint_validate), stateful_metrics=metrics_names)

        print('')
        print('#########')
        print('Validation')
        print('#########')

        # Validate dataset
        # for i in range(0, len(x_data_keypoint_validate) // batch_size):
        for i in range(0, len(x_data_keypoint_validate) // validation_batch_size):
            X = x_data_val[i * 1:min(len(x_data_val), (i + 1) * validation_batch_size)]
            y = y_data_val[i * 1:min(len(y_data_val), (i + 1) * validation_batch_size)]

            X_len = x_len_val[i * 1:min(len(x_len_val), (i + 1) * validation_batch_size)]

            X_keypoint = x_data_keypoint_validate[
                         i * 1:min(len(x_data_keypoint_validate), (i + 1) * validation_batch_size)]

            x_list = []
            y_list = []

            x_len_list = []
            y_len_list = []

            Y_zeros_list = []
            x_key_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                # x_seq, y_seq = sentence_sequence_generator_npz(X[i_data], y[i_data])

                load_npz = np.load(X[i_data])

                x_npz = load_npz['arr_0']

                load_npz = np.pad(x_npz, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]

                x_npy = x_npy.reshape((x_npy.shape[0], 1, 27, 3))
                x_npy = np.pad(x_npy, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))
                x_key_list.append(np.asarray(x_npy))

                x_list.append(load_npz)

                y_list.append(y[i_data])

            # print(np.array(x_list).shape)
            # print(np.array(x_key_list).shape)
            # print(np.array(y_list).shape)
            # print(np.array(y_list))

            predict = network.test_on_batch(x=[np.array(x_list), np.array(x_key_list)], y=np.array(y_list))

            values = [('loss', predict[0]), ('accuracy', predict[1])]

            acc.append(predict[1])
            loss.append(predict[0])
            pb_val.add(validation_batch_size, values=values)

        acc_avg = np.average(acc)
        loss_avg = np.average(loss)

        if acc_avg > acc_:
            acc_ = acc_avg
            network.save(f'{MODEL_SAVE_PATH}/LSA64_acc_{acc_}_epoch_{epoch}.h5')

        print('')
        print('#########')
        print(f'LOSS AVG / EPOCH {epoch + 1} : {loss_avg}')
        print(f'ACC AVG / EPOCH {epoch + 1} : {acc_avg}')
        print('#########')


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

    network.save(f'{MODEL_SAVE_PATH}/LSA64_acc_{acc_}_epoch_{epochs}.h5')

    # print(acc_model)
    # print(loss_model)
    #
    # error_avg = np.average(error_model)
    # print('#########')
    # print(f'model WER : {error_avg}')
    # print('######### xoxo #########')
    # print('train done')


def train_single():
    x_data, x_data_val, y_data, y_data_val, x_data_keypoint, x_data_keypoint_validate, x_len, x_len_val, max_len = generate_data_list()

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(max_len, 56, 56, 256))

    # Input Keypoint
    i_keypoint = tf.keras.Input(name="input_1_keypoint", shape=(max_len, 1, 27, 3))
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

    flatten = Flatten()(block2)  # using flatten to sync the network size - disabled if 'TMC FULL'

    # Added 2 dense layer

    dense_concat = Dense(1024,  activation='relu')(flatten)

    dense_concat = Dense(256,  activation='relu')(dense_concat)

    dense = Dense(TOTAL_CLASS, activation='softmax')(dense_concat)

    network = Model(inputs=[i_vgg], outputs=dense)

    if LOAD_MODEL:
        print('Load previous training')
        network = tf.keras.models.load_model(f'{MODEL_SAVE_PATH}LSA64_acc_0.609375_epoch_20.h5')

    network.summary()

    metrics = ['accuracy']

    network.compile(optimizer=Adam(lr=0.0001,
                                   beta_1=0.96,
                                   beta_2=0.999
                                   ), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

    if DEBUG:
        print(np.asarray(x_data).shape)
        print(np.asarray(x_data_keypoint).shape)
        print(np.asarray(y_data).shape)

    batch_size = BATCH_SIZE
    validation_batch_size = VALIDATION_BATCH_SIZE
    epochs = EPOCH

    loss_ = 999999999

    acc_model = []
    loss_model = []
    error_model = []

    metrics_names = ['loss', 'accuracy']

    acc_ = 0

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch + 1}')

        acc = []
        loss = []
        wers = []

        pb_i = Progbar(len(x_data), stateful_metrics=metrics_names)

        for i in range(0, len(x_data) // batch_size):
            X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
            y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]

            X_keypoint = x_data_keypoint[i * batch_size:min(len(x_data_keypoint), (i + 1) * batch_size)]

            X_len = x_len[i * batch_size:min(len(x_len), (i + 1) * batch_size)]

            x_list = []
            y_list = []
            x_len_list = []
            x_key_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                # x_seq, y_seq = sentence_sequence_generator_npz(X[i_data], y[i_data])

                load_npz = np.load(X[i_data])

                if DEBUG:
                    print(X[i_data][0])
                    print(X_keypoint[i_data])

                x_npz = load_npz['arr_0']

                load_npz = np.pad(x_npz, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]

                x_npy = x_npy.reshape((x_npy.shape[0], 1, 27, 3))
                x_npy = np.pad(x_npy, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))

                x_key_list.append(np.asarray(x_npy))
                x_list.append(load_npz)

                y_list.append(y[i_data])

                if DEBUG:
                    print(x_npy.shape)
                    print(load_npz.shape)

            if DEBUG:
                print(np.array(x_list).shape)
                print(np.array(x_key_list).shape)

            history = network.train_on_batch(x=[np.array(x_list)], y=np.array(y_list))

            # history = network.fit(
            #     x=[np.array(x_list), np.array(y_list), np.array(x_len_list), np.array(y_len_list)], y=np.array(Y_zeros_list), batch_size=1)

            # print(f'history : {history}')

            values = [('loss', history[0]),('accuracy', history[1])]

            pb_i.add(batch_size, values=values)

            # acc.append(history[1])
            # loss.append(history[0])

            # print(f'Loss : {X[0]}')
            # print(X)

        pb_val = Progbar(len(x_data_keypoint_validate), stateful_metrics=metrics_names)

        print('')
        print('#########')
        print('Validation')
        print('#########')

        # Validate dataset
        # for i in range(0, len(x_data_keypoint_validate) // batch_size):
        for i in range(0, len(x_data_keypoint_validate) // validation_batch_size):
            X = x_data_val[i * 1:min(len(x_data_val), (i + 1) * validation_batch_size)]
            y = y_data_val[i * 1:min(len(y_data_val), (i + 1) * validation_batch_size)]

            X_len = x_len_val[i * 1:min(len(x_len_val), (i + 1) * validation_batch_size)]

            X_keypoint = x_data_keypoint_validate[
                         i * 1:min(len(x_data_keypoint_validate), (i + 1) * validation_batch_size)]

            x_list = []
            y_list = []

            x_len_list = []
            y_len_list = []

            Y_zeros_list = []
            x_key_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                # x_seq, y_seq = sentence_sequence_generator_npz(X[i_data], y[i_data])

                load_npz = np.load(X[i_data])

                x_npz = load_npz['arr_0']

                load_npz = np.pad(x_npz, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]

                x_npy = x_npy.reshape((x_npy.shape[0], 1, 27, 3))
                x_npy = np.pad(x_npy, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))
                x_key_list.append(np.asarray(x_npy))

                x_list.append(load_npz)

                y_list.append(y[i_data])

            # print(np.array(x_list).shape)
            # print(np.array(x_key_list).shape)
            # print(np.array(y_list).shape)
            # print(np.array(y_list))

            predict = network.test_on_batch(x=[np.array(x_list)], y=np.array(y_list))

            values = [('loss', predict[0]), ('accuracy', predict[1])]

            acc.append(predict[1])
            loss.append(predict[0])
            pb_val.add(validation_batch_size, values=values)

        acc_avg = np.average(acc)
        loss_avg = np.average(loss)

        if acc_avg > acc_:
            acc_ = acc_avg
            network.save(f'{MODEL_SAVE_PATH}/LSA64_acc_{acc_}_epoch_{epoch}.h5')

        print('')
        print('#########')
        print(f'LOSS AVG / EPOCH {epoch + 1} : {loss_avg}')
        print(f'ACC AVG / EPOCH {epoch + 1} : {acc_avg}')
        print('#########')


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

    network.save(f'{MODEL_SAVE_PATH}/LSA64_single_acc_{acc_}_epoch_{epochs}.h5')

    # print(acc_model)
    # print(loss_model)
    #
    # error_avg = np.average(error_model)
    # print('#########')
    # print(f'model WER : {error_avg}')
    # print('######### xoxo #########')
    # print('train done')


if __name__ == '__main__':
    '''
    Classify LSA 64
    '''
    # x_data, y_data = generate_data_list()
    # get_value(x_data)

    # convert_data()

    # train_multi()

    train_single()

    # generate_data_list()
