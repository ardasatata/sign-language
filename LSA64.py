import glob
import math
import os
from os import listdir
from os.path import isfile, join
from jiwer import wer

import gc

# import livelossplot
# import livelossplot as livelossplot
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
from train_custom import VGG, TCN_layer, VGG_2

from keras.models import load_model

from tensorflow.keras.optimizers import Adam, SGD

from keras_ctcmodel.CTCModel import CTCModel as CTCModel

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# plot_losses = livelossplot.PlotLossesKeras()

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

MODEL_SAVE_PATH = r'C:\Users\minelab\dev\TSL\model\LSA64\\'
CTC_MODEL_PATH = r'C:\Users\minelab\dev\TSL\model\LSA64\CTC\\'

DIR = r'F:\Arda\LSA64\LSA64_Cropped'

# DIR = r'E:\arda\LSA_64'

SAVE_PATH = r'D:\LSA64\LSA64_Cropped\\'
NPZ_DIR = r'D:\LSA64\VGG_out_new'

ORIGINAL_VIDEO_PATH = r'D:\LSA64\LSA64_Cropped\\'

# 10 Class with full sample & subject
START_FROM = 48
TOTAL_CLASS = 64

TOTAL_SUBJECT = 10
TOTAL_SAMPLE = 5

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

EPOCH = 25

# CTC Config

# Train Phase 1
# SENTENCE_ARRAY = [
#     [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],  # 5 sample
#     [12, 16, 28, 26, 24, 0, 0, 0, 0, 0],
#     [9, 16, 7, 8, 4, 0, 0, 0, 0, 0],
#     [12, 64, 28, 26, 24, 0, 0, 0, 0, 0],
#     [42, 12, 64, 26, 24, 0, 0, 0, 0, 0],
#     [12, 32, 7, 3, 4, 0, 0, 0, 0, 0],
#     [7, 3, 4, 5, 53, 0, 0, 0, 0, 0],
#     [9, 64, 28, 26, 24, 0, 0, 0, 0, 0],
#     [12, 16, 7, 8, 4, 0, 0, 0, 0, 0],
#     [12, 16, 28, 26, 25, 0, 0, 0, 0, 0],
# ]

# Train Phase 2
SENTENCE_ARRAY = [
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],  # 5 sample
    [12, 16, 28, 26, 24, 0, 0, 0, 0, 0],
    [9, 16, 7, 8, 4, 0, 0, 0, 0, 0],
    [12, 64, 28, 26, 24, 0, 0, 0, 0, 0],
    [42, 12, 64, 26, 24, 0, 0, 0, 0, 0],
    [12, 32, 7, 3, 4, 0, 0, 0, 0, 0],
    [7, 3, 4, 5, 53, 0, 0, 0, 0, 0],
    [9, 64, 28, 26, 24, 0, 0, 0, 0, 0],
    [12, 16, 7, 8, 4, 0, 0, 0, 0, 0],
    [12, 16, 28, 26, 25, 0, 0, 0, 0, 0],
    # man learn colors Yellow Bright
    [12, 16, 7, 4, 5, 0, 0, 0, 0, 0],
    # women learn green yellow light-blue
    [9, 16, 3, 4, 6, 0, 0, 0, 0, 0],
    # women buy breakfast food barbecue
    [9, 59, 31, 23, 45, 0, 0, 0, 0, 0],
    # man buy breakfast food barbecue
    [12, 59, 31, 23, 45, 0, 0, 0, 0, 0],
    # women buy breakfast food spaghetti
    [9, 59, 31, 23, 48, 0, 0, 0, 0, 0],
    # women buy breakfast food spaghetti
    [12, 59, 31, 23, 48, 0, 0, 0, 0, 0],
    # hungry man buy food barbecue
    [33, 12, 59, 23, 45, 0, 0, 0, 0, 0],
    # hungry women buy breakfast food
    [33, 9, 59, 31, 23, 0, 0, 0, 0, 0],
    # man call women buy food
    [12, 17, 9, 59, 31, 0, 0, 0, 0, 0],
    # man call women buy food
    [9, 17, 12, 59, 31, 0, 0, 0, 0, 0],
]


UNIQUE_LABEL = np.unique(SENTENCE_ARRAY)

TOTAL_LABEL = len(UNIQUE_LABEL)

print(UNIQUE_LABEL)
print(TOTAL_LABEL)

SENTENCE_ARRAY_TEST = [
    [12, 16, 3, 4, 5, 0, 0, 0, 0, 0],
    [9, 16, 28, 26, 24, 0, 0, 0, 0, 0],
    [12, 16, 7, 8, 4, 0, 0, 0, 0, 0],
    [9, 64, 28, 26, 24, 0, 0, 0, 0, 0],
    [42, 9, 64, 26, 24, 0, 0, 0, 0, 0],
    [9, 32, 7, 3, 4, 0, 0, 0, 0, 0],
    [7, 53, 3, 4, 5, 0, 0, 0, 0, 0],
    [9, 64, 28, 26, 24, 0, 0, 0, 0, 0],
    [9, 16, 7, 8, 4, 0, 0, 0, 0, 0],
]

# SENTENCE_ARRAY_TEST = [
#     [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],  # 5 sample
#     [12, 16, 28, 26, 24, 0, 0, 0, 0, 0],
#     [9, 16, 7, 8, 4, 0, 0, 0, 0, 0],
#     [12, 64, 28, 26, 24, 0, 0, 0, 0, 0],
#     [42, 12, 64, 26, 24, 0, 0, 0, 0, 0],
#     [12, 32, 7, 3, 4, 0, 0, 0, 0, 0],
#     [7, 3, 4, 5, 53, 0, 0, 0, 0, 0],
#     [9, 64, 28, 26, 24, 0, 0, 0, 0, 0],
#     [12, 16, 7, 8, 4, 0, 0, 0, 0, 0],
#     [12, 16, 28, 26, 25, 0, 0, 0, 0, 0],
# ]

# SENTENCE_ARRAY_TEST = [
#     # man learn colors Yellow Bright
#     [12, 16, 7, 4, 5, 0, 0, 0, 0, 0],
#     # women learn green yellow light-blue
#     [9, 16, 3, 4, 6, 0, 0, 0, 0, 0],
#     # women buy breakfast food barbecue
#     [9, 59, 31, 23, 45, 0, 0, 0, 0, 0],
#     # man buy breakfast food barbecue
#     [12, 59, 31, 23, 45, 0, 0, 0, 0, 0],
#     # women buy breakfast food spaghetti
#     [9, 59, 31, 23, 48, 0, 0, 0, 0, 0],
#     # women buy breakfast food spaghetti
#     [12, 59, 31, 23, 48, 0, 0, 0, 0, 0],
#     # hungry man buy food barbecue
#     [33, 12, 59, 23, 45, 0, 0, 0, 0, 0],
#     # hungry women buy breakfast food
#     [33, 9, 59, 31, 23, 0, 0, 0, 0, 0],
#     # man call women buy food
#     [12, 17, 9, 59, 31, 0, 0, 0, 0, 0],
#     # man call women buy food
#     [9, 17, 12, 59, 31, 0, 0, 0, 0, 0],
# ]

TOTAL_LABEL_TEST = len(np.unique(SENTENCE_ARRAY_TEST))

print(TOTAL_LABEL_TEST)

WORD_COUNT = 5
LABEL_LENGTH = 10

SENTENCE_SAMPLE = 1


def y_generator(class_num=0, total_class=2):
    arr = []
    for i in range(0, total_class):
        if i == class_num:
            arr.append(1)
        else:
            arr.append(0)

    return arr


def generate_data_list():
    print("Generate Dataset")

    x_data = []
    y_data = []

    for class_num in range(START_FROM, TOTAL_CLASS):
        for subject_num in range(0, TOTAL_SUBJECT):
            for sample_num in range(0, TOTAL_SAMPLE):
                file = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                    .format(dir=DIR, class_num=str(class_num + 1).zfill(3), subject_num=str(subject_num + 1).zfill(3),
                            sample_num=str(sample_num + 1).zfill(3), ext='avi')
                print(file)

                data = []

                try:
                    cap = cv2.VideoCapture(file)

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            resized_image = cv2.resize(frame, (RESOLUTION, RESOLUTION))
                            # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                            #
                            # back2rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                            # data.append(np.fliplr(resized_image))
                            data.append(resized_image)
                        else:
                            break
                except cv2.error as e:
                    print(e)
                    False

                    cap.release()

                # print(np.asarray(data))

                # print(np.asarray(data).shape)
                # print(class_num)
                x_data.append(data)
                y_data.append(class_num)

    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)

    padded_x = tf.keras.preprocessing.sequence.pad_sequences(x_data, value=0, padding='post', maxlen=MAX_FRAME)
    print(np.asarray(padded_x).shape)

    # print(x_data)
    print(y_data)

    return padded_x, y_data


START_FROM_CLASS = 10


def generate_data_list_npz(loadNPZ=False):
    print("Generate Dataset from npz")

    x_data = []
    y_data = []

    for class_num in range(0, TOTAL_CLASS):
        for subject_num in range(0, TOTAL_SUBJECT):
            for sample_num in range(0, TOTAL_SAMPLE):
                file = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                    .format(dir=NPZ_DIR, class_num=str(class_num + 1).zfill(3),
                            subject_num=str(subject_num + 1).zfill(3),
                            sample_num=str(sample_num + 1).zfill(3), ext='npz')
                print(file)

                if loadNPZ:
                    x_npz = np.load(file)
                    print(x_npz['arr_0'].shape)
                    x_data.append(np.asarray(x_npz['arr_0']))
                    # print(np.asarray(data))
                    # print(np.asarray(data).shape)
                    # print(class_num)
                else:
                    x_data.append(file)

                y_data.append(y_generator(class_num, TOTAL_CLASS))

    print(x_data)
    print(y_data)

    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)

    return x_data, y_data


def sentence_dataset_generator(sentence_array=None, sample_number=SENTENCE_SAMPLE, subject_num=None, sample_num=None):
    if subject_num is None:
        subject_num = [1, 8]
    if sample_num is None:
        sample_num = [1, 3]
    if sentence_array is None:
        sentence_array = SENTENCE_ARRAY

    print("Generate Sentence from NPZ")

    sentences_array = []
    words_annotation = []

    for sentence in range(0, len(sentence_array)):
        samples = []
        for sample in range(0, sample_number):
            words = []
            random_subject = random.randint(subject_num[0], subject_num[1])
            for word in range(0, LABEL_LENGTH):
                if sentence_array[sentence][word] == 0:
                    # print('skip')
                    continue
                file = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                    .format(dir=NPZ_DIR, class_num=str(sentence_array[sentence][word]).zfill(3),
                            subject_num=str(random_subject).zfill(3),
                            sample_num=str(random.randint(sample_num[0], sample_num[1])).zfill(3), ext='npz')
                # print(file)
                words.append(file)
            words_annotation.append(sentence_array[sentence])
            samples.append(words)

        sentences_array.append(samples)
        # print(samples)

    x_len = []
    y_len = []

    for len_ in range(0, len(sentence_array) * LABEL_LENGTH * sample_number):
        # x_len.append(MAX_FRAME * 5)
        x_len.append(20)
        y_len.append(LABEL_LENGTH)

    # print(np.asarray(sentences_array).shape)

    x_data = np.reshape(sentences_array, (len(sentence_array) * sample_number, WORD_COUNT))
    y_data = np.asarray(words_annotation)

    # print(x_data.shape)
    # print(np.asarray(words_annotation).shape)
    #
    # print(x_data)
    # print(words_annotation)
    #
    # print(x_len)
    # print(y_len)

    return x_data, y_data, x_len, y_len


def sentence_sequence_generator_npz(x, y):
    """ Generate word sequence from filename .npz """
    x_data = []

    for i_data in range(0, len(x)):
        x_npz = np.load(x[i_data])
        # print(x_npz['arr_0'])

        x_skipped = np.asarray(x_npz['arr_0'][::8, :, :, :])

        # print(x_skipped.shape)
        # exit(0)

        # x_data.append(np.asarray(x_npz['arr_0']))
        x_data.append(x_skipped)

    # x_data = np.reshape(x_data, (700, 56, 56, 256))
    x_data = np.reshape(x_data, (18 * WORD_COUNT, 56, 56, 256))

    # print(x_data.shape)
    # exit(0)

    return x_data, y


def train(shuffle=True):
    x_data, y_data = generate_data_list_npz()

    if shuffle:
        c = list(zip(x_data, y_data))
        random.shuffle(c)
        x_data, y_data = zip(*c)

    print(x_data)
    print(y_data)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(140, 56, 56, 256))

    vgg = VGG(i_vgg)
    m_vgg = Model(inputs=[i_vgg], outputs=[vgg])

    # LSTM
    # lstm = LSTM(64, return_sequences=True)(vgg)
    # lstm = LSTM(64, return_sequences=True)(lstm)F
    # lstm = Flatten()(lstm)
    # lstm = Dense(512, activation='relu')(lstm)
    # lstm = Dropout(0.5)(lstm)
    # lstm = Dense(256, activation='relu')(lstm)
    # lstm = Dropout(0.25)(lstm)
    # lstm = Dense(TOTAL_CLASS, activation='softmax')(lstm)
    # tcn_model = Model(inputs=[i_vgg], outputs=[lstm])

    # TCN

    '''
    TCN -> Dense
    '''

    o_tcn_full = TCN_layer(vgg, 5)
    # global_pool = GlobalAveragePooling2D(name="global_max_full")(o_tcn_full)
    # flatten = Flatten()(o_tcn_full) # using flatten to sync the network size - disabled if 'TMC FULL'
    dense = Dense(256, activation='relu', name='dense_o_tcn1')(o_tcn_full)

    '''
    TMC (cont)
    '''

    o_tcn_block1 = TCN_layer(dense, 1)
    o_tcn_block1 = Dense(256, name='dense_o_tcn_intra_block1')(o_tcn_block1)
    o_tcn_block1 = Dense(256)(o_tcn_block1)
    block1 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block1)

    i_tcn2 = block1
    o_tcn2 = TCN_layer(i_tcn2, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2 = Dense(256, activation='relu', name='dense_o_tcn2')(o_tcn2)
    o_tcn_block2 = TCN_layer(dense2, 1)
    o_tcn_block2 = Dense(256, name='dense_o_tcn_intra_block2')(o_tcn_block2)
    o_tcn_block2 = Dense(256)(o_tcn_block2)
    block2 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block2)

    '''
    TMC (cont) # endregion
    '''

    # classification
    flatten = Flatten()(block2)  # using flatten to sync the network size
    dense = Dense(256, activation='relu')(flatten)
    # endregion

    # classification total class - dense
    dense = Dense(TOTAL_CLASS, activation='softmax')(dense)
    tcn_model = Model(inputs=[i_vgg], outputs=[dense])

    tcn_model.summary()
    # exit(0)

    # # Full Frame Model
    # m_vgg.summary()

    ### The Custom Loop
    # The train_on_batch function
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(0.0001)
    # optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)

    # Compile
    # m_vgg.compile(loss=loss, optimizer=optimizer)

    metrics_names = ['acc']

    tcn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics_names)

    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)

    batch_size = 5
    epochs = EPOCH

    loss_ = 999999999

    acc_model = []
    loss_model = []

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch}')

        acc = []
        loss = []

        pb_i = Progbar(TOTAL_CLASS * TOTAL_SUBJECT * TOTAL_SAMPLE, stateful_metrics=metrics_names)

        for i in range(0, len(x_data) // batch_size):
            X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
            y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]

            x_list = []
            y_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                # print(len(X))
                # print(X[i_data])
                # print(y[i_data])
                x_npz = np.load(X[i_data])
                # print(x_npz['arr_0'].shape)

                x_list.append(np.asarray(x_npz['arr_0']))
                y_list.append(y[i_data])

            X = tcn_model.train_on_batch(np.array(x_list), np.array(y_list))

            values = [('acc', X[1])]

            pb_i.add(batch_size, values=values)

            acc.append(X[1])
            loss.append(X[0])

            # print(f'Loss : {X[0]}')
            # print(X)

        print(f'Loss : {np.average(np.array(loss))}, Accuracy : {np.average(np.array(acc))}')

        acc_model.append(np.average(np.array(acc)))
        loss_model.append(np.average(np.array(loss)))

        if np.average(np.array(loss)) < loss_:
            loss_ = np.average(np.array(loss))
            tcn_model.save(
                filepath=f'{MODEL_SAVE_PATH}{"TMC_Classification_LSA64"}_{EPOCH}_{TOTAL_CLASS}_{TOTAL_SUBJECT}_{TOTAL_SAMPLE}_{loss_}.h5')

        # val_loss = []
        # for i in range(0, len(x_data) // batch_size):
        #     X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
        #     y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]
        #     val_loss.append(tcn_model.validate_on_batch(X, y))
        #
        # print('Validation Loss: ' + str(np.mean(val_loss)))

    tcn_model.save(
        filepath=f'{MODEL_SAVE_PATH}{"TMC_Classification_LSA64"}_{EPOCH}_{TOTAL_CLASS}_{TOTAL_SUBJECT}_{TOTAL_SAMPLE}.h5')

    print(acc_model)
    print(loss_model)

    print('train')


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def get_value(x_data):  # Generate NPZ from weight model

    filename = []

    for class_num in range(START_FROM, TOTAL_CLASS):
        for subject_num in range(0, TOTAL_SUBJECT):
            for sample_num in range(0, TOTAL_SAMPLE):
                file = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                    .format(dir=EXTRACT_DESTINATION, class_num=str(class_num + 1).zfill(3),
                            subject_num=str(subject_num + 1).zfill(3),
                            sample_num=str(sample_num + 1).zfill(3), ext='npz')
                print(file)
                filename.append(file)

    folders = [f.path for f in os.scandir(DIR) if f.is_dir()]
    onlyfiles = [f for f in listdir(DIR) if isfile(join(DIR, f))]

    print(onlyfiles)

    filePaths = glob.glob(DIR + "/*")

    for i in range(0, len(x_data)):
        # Save NPZ disini
        try:

            output = get_output_layer(src=np.asarray(x_data[i]))
            tf.keras.backend.clear_session()
            gc.collect()
            print(output.shape)
            print(filename[i])

            savez_compressed(filename[i], output)

        except cv2.error as e:
            print(e)
            False

        tf.keras.backend.clear_session()

        print('save npz')


def crop_dataset():
    print("Crop Dataset")

    for class_num in range(START_FROM, TOTAL_CLASS):
        for subject_num in range(0, TOTAL_SUBJECT):
            file = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                .format(dir=DIR,
                        class_num=str(class_num + 1).zfill(
                            3),
                        subject_num=str(
                            subject_num + 1).zfill(3),
                        sample_num=str('1').zfill(3),
                        ext=EXT)
            print(file)

            cap = cv2.VideoCapture(file)

            # cap.set(1, 35)
            _ret, _frame = cap.read()

            # print(file[FILENAME_PADDING:-4])

            _frame = np.asarray(_frame)
            nose_x, nose_y = get_position(_frame.astype(np.uint8))

            for sample_num in range(0, TOTAL_SAMPLE):
                file = r"{dir}\{class_num}_{subject_num}_{sample_num}.{ext}" \
                    .format(dir=DIR,
                            class_num=str(class_num + 1).zfill(
                                3),
                            subject_num=str(
                                subject_num + 1).zfill(3),
                            sample_num=str(
                                sample_num + 1).zfill(3),
                            ext=EXT)
                print(file)

                try:
                    cap = cv2.VideoCapture(file)

                    # cap.set(1, 1)
                    ret, frame = cap.read()

                    print(file[FILENAME_PADDING:-4])

                    # frame = np.asarray(frame)
                    #
                    # nose_x, nose_y = get_position(frame.astype(np.uint8))

                    crop_video(file, file[FILENAME_PADDING:-4] + '.avi', nose_x, nose_y)

                    cap.release()

                    tf.keras.backend.clear_session()
                    gc.collect()

                except cv2.error as e:
                    print(e.msg)
                    False


def crop_location(x_pos, y_pos):
    X1 = 610
    Y1 = 120
    Y2 = 420

    XPOS = int(x_pos)
    YPOS = int(y_pos)

    # forced crop

    XPOS = 882
    YPOS = 264

    x1 = XPOS - X1
    x2 = XPOS + X1
    y1 = YPOS - Y1
    y2 = YPOS + Y2

    print(x2 - x1)
    print(x2)
    print(x1)

    return x1, x2, 0, 1080, x2 - x1, 1080


def get_position(file):
    # For static images:
    pose = mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.4)

    # image = cv2.imread(file)
    image = file
    image_hight, image_width, _ = image.shape

    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
    )
    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # cv2.imwrite(DIR3 + 'annotated_image_' + filename + '.png', annotated_image)
    pose.close()

    nose_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width
    nose_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight

    return nose_x, nose_y


def crop_video(file, filename='test1.avi', x=640, y=360):
    print('cropping')
    cap = cv2.VideoCapture(file)
    # cap.set(1, 2)
    cap.set(cv2.CAP_PROP_FPS, 30)

    x1, x2, y1, y2, width, height = crop_location(x, y)

    outcrop = cv2.VideoWriter(SAVE_PATH + filename, cv2.VideoWriter_fourcc(*'XVID'), 60, (width, height))

    # print(width, height)
    print(file)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Y , X
            cropped = frame[y1:y2, x1:x2]
            outcrop.write(cropped)
        else:
            break

    cap.release()
    outcrop.release()
    cv2.destroyAllWindows()


import matplotlib.pyplot as plt


def model_predict(filename="025_009_003.npz"):
    model = load_model(f'{MODEL_SAVE_PATH}{"TMC_Classification_LSA64_15_64_10_5_0.15316064515460312.h5"}')

    model.summary()

    x_npz = np.load(f'{EXTRACT_DESTINATION}{"025_009_003.npz"}')
    x_npz = x_npz['arr_0']

    x_val = np.reshape(x_npz, (1, 140, 56, 56, 256))

    result = model.predict(x_val)

    norm = np.linalg.norm(result[0])

    print(np.argmax(result))

    print(result[0] / norm)

    for i in result[0] / norm:
        print(i)

    # PLOT

    plt.style.use('ggplot')

    class_list = range(0, TOTAL_CLASS)
    predict_result = result[0] / norm

    x_pos = [i for i, _ in enumerate(class_list)]

    plt.bar(x_pos, predict_result, color='green')

    plt.xticks(class_list)
    plt.xlabel("Class")
    plt.ylabel("")
    plt.title("Predict")

    plt.show()


def predict_ctc():
    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(MAX_FRAME * WORD_COUNT, 56, 56, 256))

    vgg = VGG_2(i_vgg)

    # TCN

    '''
    TCN -> Dense
    '''

    o_tcn_full = TCN_layer(vgg, 5)
    # global_pool = GlobalAveragePooling2D(name="global_max_full")(o_tcn_full)
    # flatten = Flatten()(o_tcn_full) # using flatten to sync the network size - disabled if 'TMC FULL'
    dense = Dense(256, name='dense_o_tcn1')(o_tcn_full)

    '''
    TMC (cont)
    '''

    o_tcn_block1 = TCN_layer(dense, 1)
    o_tcn_block1 = Dense(256, name='dense_o_tcn_intra_block1')(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    block1 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block1)

    i_tcn2 = block1
    o_tcn2 = TCN_layer(i_tcn2, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2 = Dense(256, name='dense_o_tcn2')(o_tcn2)
    o_tcn_block2 = TCN_layer(dense2, 1)
    o_tcn_block2 = Dense(512, name='dense_o_tcn_intra_block2')(o_tcn_block2)
    o_tcn_block2 = Dense(512)(o_tcn_block2)
    block2 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block2)

    '''
    TMC (cont) # endregion
    '''

    '''
    Sequence Learning
    '''

    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(block2)
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    dense = TimeDistributed(Dense(64 + 2, name="dense"))(blstm)  # -- 2
    outrnn = Activation('softmax', name='softmax')(dense)

    '''
    Sequence Learning # endregion
    '''

    network = CTCModel([i_vgg], [outrnn])  # -- 4

    network.compile(tf.keras.optimizers.Adam(0.00001))

    # network.load_weights(file_weights=CTC_MODEL_PATH + 'model_weights.hdf5', by_name=True)

    # network.load_model(path_dir='', optimizer=Adam(lr=1e-4), init_archi=True)
    network.load_model(path_dir=r"C:\Users\minelab\dev\TSL\model\LSA64\\", file_weights='/model_weights.hdf5',
                       optimizer=Adam(0.00001), init_last_layer=False, init_archi=False)
    # network.load_model(path_dir=r"C:\Users\minelab\dev\TSL\model\LSA64\\", file_weights='/model_weights.hdf5')

    network.summary()

    xd, yd, xl, yl = sentence_dataset_generator(sample_number=1, sentence_array=SENTENCE_ARRAY_TEST)

    # Predict ALL
    test_data = []
    for i in range(0, len(SENTENCE_ARRAY_TEST)):
        xs, ys = sentence_sequence_generator_npz(xd[i], yd[i])
        test_data.append(xs)

    label_length = np.full(shape=len(SENTENCE_ARRAY_TEST), fill_value=5, dtype=np.int)

    result = network.predict(x=[np.array(test_data), label_length])
    words = np.where(result != -1)
    print("word", words)
    print(result)
    print(np.asarray(result[0]).shape)

    calculate_wer(result=result, gt=yd)

    print('===== Predict one by one =====\n')
    # Predict one by one
    for i in range(0, len(SENTENCE_ARRAY_TEST)):
        xs, ys = sentence_sequence_generator_npz(xd[i], yd[i])

        print('GT : ', ys)

        length = [5, 5]
        x_data = [xs, xs]

        result = network.predict(x=[np.asarray(x_data), np.asarray(length)])
        print(result[0])
        print(yd[i])

        calculate_wer(result=[result[0]], gt=[yd[i]])

    # print(np.array(test_data).shape)

    # error = wer(np.char('%d', yd), np.char('%d', result))

    print('train')


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.00005
    drop = 0.25
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def train_ctc_fit(shuffle=True):
    x_data, y_data, x_len, y_len = sentence_dataset_generator()

    if shuffle:
        c = list(zip(x_data, y_data))
        random.shuffle(c)
        x_data, y_data = zip(*c)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(MAX_FRAME * WORD_COUNT, 56, 56, 256))

    vgg = VGG_2(i_vgg)

    # TCN

    '''
    TCN -> Dense
    '''

    o_tcn_full = TCN_layer(vgg, 5)
    # global_pool = GlobalAveragePooling2D(name="global_max_full")(o_tcn_full)
    # flatten = Flatten()(o_tcn_full) # using flatten to sync the network size - disabled if 'TMC FULL'
    dense = Dense(256, name='dense_o_tcn1')(o_tcn_full)

    '''
    TMC (cont)
    '''

    o_tcn_block1 = TCN_layer(dense, 1)
    o_tcn_block1 = Dense(256, name='dense_o_tcn_intra_block1')(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    block1 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block1)

    i_tcn2 = block1
    o_tcn2 = TCN_layer(i_tcn2, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2 = Dense(256, name='dense_o_tcn2')(o_tcn2)
    o_tcn_block2 = TCN_layer(dense2, 1)
    o_tcn_block2 = Dense(512, name='dense_o_tcn_intra_block2')(o_tcn_block2)
    o_tcn_block2 = Dense(512)(o_tcn_block2)
    block2 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block2)

    '''
    TMC (cont) # endregion
    '''

    '''
    Sequence Learning
    '''

    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(block2)
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    dense = TimeDistributed(Dense(64 + 2, name="dense"))(blstm)  # -- 2
    outrnn = Activation('softmax', name='softmax')(dense)

    '''
    Sequence Learning # endregion
    '''

    network = CTCModel([i_vgg], [outrnn])  # -- 4

    network.compile(tf.keras.optimizers.Adam(0.00001))

    network.summary()

    # Load Previous Train
    if os.path.exists(r'C:\Users\minelab\dev\TSL\model\LSA64\model_weights.hdf5'):
        # network.load_weights(r'C:\Users\minelab\dev\TSL\model\LSA64\model_weights.hdf5', by_name=True)
        print('Load previous model')
        network.load_model(path_dir=r"C:\Users\minelab\dev\TSL\model\LSA64\\", file_weights='/model_weights.hdf5',
                           optimizer=Adam(0.00001), init_last_layer=False, init_archi=False)
        print('model loaded.')

    x_all = []
    y_all = []
    x_len_all = []
    y_len_all = []

    y_zeros = np.zeros(SENTENCE_SAMPLE * len(SENTENCE_ARRAY))

    Y_zeros_list = []

    for i_data in range(0, len(x_data)):
        x_seq, y_seq = sentence_sequence_generator_npz(x_data[i_data], y_data[i_data])

        x_all.append(np.asarray(x_seq))
        y_all.append(np.asarray(y_seq))

        x_len_all.append(np.asarray(x_len[i_data]))
        y_len_all.append(np.asarray(y_len[i_data]))

        Y_zeros_list.append(y_zeros[i_data])

    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    network.fit(x=[np.array(x_all), np.array(y_all), np.array(x_len_all), np.array(y_len_all)],
                y=np.array(Y_zeros_list), epochs=EPOCH, batch_size=1,
                validation_split=0.2, callbacks=callbacks_list)

    network.save_model(path_dir=f'{MODEL_SAVE_PATH}')

    xd, yd, xl, yl = sentence_dataset_generator(sample_number=1)

    print(xd)
    print(yd)

    test_data = []

    for i in range(0, len(SENTENCE_ARRAY)):
        xs, ys = sentence_sequence_generator_npz(xd[i], yd[i])
        test_data.append(xs)

    # print(np.array(test_data).shape)

    label_length = np.full(shape=len(SENTENCE_ARRAY), fill_value=5, dtype=np.int)

    result = network.predict(x=[np.array(test_data), label_length])
    words = np.where(result != -1)
    print("word", words)
    print(result)
    print(np.asarray(result[0]).shape)

    calculate_wer(result=result, gt=yd)

    # error = wer(np.char('%d', yd), np.char('%d', result))

    print('train done')


def train_ctc(shuffle=True):
    x_data, y_data, x_len, y_len = sentence_dataset_generator()

    if shuffle:
        c = list(zip(x_data, y_data))
        random.shuffle(c)
        x_data, y_data = zip(*c)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(MAX_FRAME * WORD_COUNT, 56, 56, 256))

    vgg = VGG_2(i_vgg)
    # m_vgg = Model(inputs=[i_vgg], outputs=[vgg])

    # TCN

    '''
    TCN -> Dense
    '''

    o_tcn_full = TCN_layer(vgg, 5)
    # global_pool = GlobalAveragePooling2D(name="global_max_full")(o_tcn_full)
    # flatten = Flatten()(o_tcn_full) # using flatten to sync the network size - disabled if 'TMC FULL'
    dense = Dense(256, name='dense_o_tcn1')(o_tcn_full)

    '''
    TMC (cont)
    '''

    o_tcn_block1 = TCN_layer(dense, 1)
    o_tcn_block1 = Dense(256, name='dense_o_tcn_intra_block1')(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    block1 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block1)

    i_tcn2 = block1
    o_tcn2 = TCN_layer(i_tcn2, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2 = Dense(256, name='dense_o_tcn2')(o_tcn2)
    o_tcn_block2 = TCN_layer(dense2, 1)
    o_tcn_block2 = Dense(512, name='dense_o_tcn_intra_block2')(o_tcn_block2)
    o_tcn_block2 = Dense(512)(o_tcn_block2)
    block2 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block2)

    '''
    TMC (cont) # endregion
    '''

    '''
    Sequence Learning
    '''

    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(block2)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    dense = TimeDistributed(Dense(TOTAL_LABEL + 1, name="dense"))(blstm)  # -- 2
    outrnn = Activation('softmax', name='softmax')(dense)

    '''
    Sequence Learning # endregion
    '''

    network = CTCModel([i_vgg], [outrnn])  # -- 4

    print(network.get_model_train())

    # network = CTCLayer(name="ctc_loss")(outrnn, i_vgg)

    network.compile(tf.keras.optimizers.Adam(0.00001))
    # network.load_model(path_dir=r"C:\Users\minelab\dev\TSL\model\LSA64", file_weights='/model_weights.hdf5',
    #                   optimizer=Adam(lr=0.0005), init_last_layer=False, init_archi=False)
    network.summary()

    metrics_names = ['val']

    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)

    batch_size = 8
    epochs = EPOCH

    loss_ = 999999999

    acc_model = []
    loss_model = []

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch}')

        acc = []
        loss = []

        pb_i = Progbar(len(SENTENCE_ARRAY) * SENTENCE_SAMPLE, stateful_metrics=metrics_names)

        y_zeros = np.zeros(SENTENCE_SAMPLE * len(SENTENCE_ARRAY))

        for i in range(0, len(x_data) // batch_size):
            X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
            y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]

            X_len = x_len[i * batch_size:min(len(x_len), (i + 1) * batch_size)]
            Y_len = y_len[i * batch_size:min(len(y_len), (i + 1) * batch_size)]

            Y_zeros = y_zeros[i * batch_size:min(len(y_zeros), (i + 1) * batch_size)]

            x_list = []
            y_list = []

            x_len_list = []
            y_len_list = []

            Y_zeros_list = []

            # print(np.asarray(X).shape)
            # print(np.asarray(y).shape)

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                x_seq, y_seq = sentence_sequence_generator_npz(X[i_data], y[i_data])

                x_list.append(np.asarray(x_seq))
                y_list.append(np.asarray(y_seq))

                x_len_list.append(np.asarray(X_len[i_data]))
                y_len_list.append(np.asarray(Y_len[i_data]))

                Y_zeros_list.append(Y_zeros[i_data])

            # print(np.array(x_list).shape)
            # print(np.array(y_list).shape)
            # print(np.array(x_list))
            # print(np.array(y_list))
            # print(np.array(x_len_list))
            # print(np.array(y_len_list))
            # print(np.array(Y_zeros_list))

            history = network.train_on_batch(
                x=[np.array(x_list), np.array(y_list), np.array(x_len_list), np.array(y_len_list)],
                y=np.array(Y_zeros_list))

            # print(f'history : {history}')

            values = [('val', history)]

            pb_i.add(batch_size, values=values)

            # acc.append(history[1])
            # loss.append(history[0])

            # print(f'Loss : {X[0]}')
            # print(X)

        # print(f'Loss : {np.average(np.array(loss))}, Accuracy : {np.average(np.array(acc))}')
        # acc_model.append(np.average(np.array(acc)))
        # loss_model.append(np.average(np.array(loss)))

        if epoch > 1:
            xd, yd, xl, yl = sentence_dataset_generator(sample_number=1)

            test_data = []

            for i in range(0, len(SENTENCE_ARRAY)):
                xs, ys = sentence_sequence_generator_npz(xd[i], yd[i])
                test_data.append(xs)

            # print(np.array(test_data).shape)

            result = network.predict(x=[np.array(test_data), np.array([5, 5, 5, 5, 5])])
            words = np.where(result != -1)
            print("word", words)
            print(result)
            print(np.asarray(result[0]).shape)

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

    network.save_model(path_dir=f'{MODEL_SAVE_PATH}')

    print(acc_model)
    print(loss_model)

    print('train')


class CustomCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        # self.model.save_model(path_dir=f'{MODEL_SAVE_PATH}')

        xd, yd, xl, yl = sentence_dataset_generator(sample_number=2)

        test_data = []

        for i in range(0, 2):
            xs, ys = sentence_sequence_generator_npz(xd[i], yd[i])
            test_data.append(xs)

        result = self.predict(x=[np.array(test_data), np.array([5, 5])])
        words = np.where(result != -1)
        print("word", words)
        print(result)
        print(np.asarray(result[0]).shape)

        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


def calculate_wer(gt=None, result=None, length=5):
    if result is None:
        result = [
            [1, 1, 1, 4],
            [1, 1, 1, 4]
        ]
    if gt is None:
        gt = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]

    gt2 = np.array(gt).astype(str)[:, 0:length]
    hp2 = np.array(result).astype(int).astype(str)[:, 0:length]
    # hp2 = np.array(result[0]).astype(int).astype(str)[:, 0:length]

    # print(gt2)
    # print(hp2)

    gt3 = []
    hp3 = []

    for i in gt2:
        string = (" ".join(i))
        # print(string)
        gt3.append(string)

    for i in hp2:
        string = (" ".join(i))
        # print(string)
        hp3.append(string)

    # print(gt3)
    # print(hp3)

    error = wer(gt3, hp3)

    # print("Word Error rate = " + str(error))

    return error

def calculate_wer(gt=None, result=None, length=5):
    if result is None:
        result = [
            [1, 1, 1, 4],
            [1, 1, 1, 4]
        ]
    if gt is None:
        gt = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]

    gt2 = np.array(gt).astype(str)[:, 0:length]
    hp2 = np.array(result).astype(int).astype(str)[:, 0:length]
    # hp2 = np.array(result[0]).astype(int).astype(str)[:, 0:length]

    # print(gt2)
    # print(hp2)

    gt3 = []
    hp3 = []

    for i in gt2:
        string = (" ".join(i))
        # print(string)
        gt3.append(string)

    for i in hp2:
        string = (" ".join(i))
        # print(string)
        hp3.append(string)


    error = wer(gt3, hp3)

    return error


def demo_testing(test_sentence, subject_num=None, sample_num=None):
    N = 5
    test_sentence = np.pad(test_sentence, (0, N), 'constant')

    test_sentence = [test_sentence, test_sentence]

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(MAX_FRAME * WORD_COUNT, 56, 56, 256))

    vgg = VGG_2(i_vgg)

    o_tcn_full = TCN_layer(vgg, 5)
    dense = Dense(256, name='dense_o_tcn1')(o_tcn_full)

    o_tcn_block1 = TCN_layer(dense, 1)
    o_tcn_block1 = Dense(256, name='dense_o_tcn_intra_block1')(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    o_tcn_block1 = Dense(512)(o_tcn_block1)
    block1 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block1)

    i_tcn2 = block1
    o_tcn2 = TCN_layer(i_tcn2, 5)

    dense2 = Dense(256, name='dense_o_tcn2')(o_tcn2)
    o_tcn_block2 = TCN_layer(dense2, 1)
    o_tcn_block2 = Dense(512, name='dense_o_tcn_intra_block2')(o_tcn_block2)
    o_tcn_block2 = Dense(512)(o_tcn_block2)
    block2 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block2)
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(block2)
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    dense = TimeDistributed(Dense(64 + 2, name="dense"))(blstm)  # -- 2
    outrnn = Activation('softmax', name='softmax')(dense)

    network = CTCModel([i_vgg], [outrnn])  # -- 4
    network.compile(tf.keras.optimizers.Adam(0.00001))

    network.load_model(path_dir=r"C:\Users\minelab\dev\TSL\model\LSA64\\", file_weights='/model_weights.hdf5',
                       optimizer=Adam(0.00001), init_last_layer=False, init_archi=False)

    network.summary()

    xd, yd, xl, yl = sentence_dataset_generator(sample_number=1, sentence_array=test_sentence)

    print('===== Predict one by one =====\n')
    # Predict one by one
    for i in range(0, 1):
        xs, ys = sentence_sequence_generator_npz(xd[i], yd[i])

        print('GT : ', ys)

        length = [5, 5]
        x_data = [xs, xs]

        result = network.predict(x=[np.asarray(x_data), np.asarray(length)])
        print(result[0])
        print(yd[i])

        calculate_wer(result=[result[0]], gt=[yd[i]])

    # Preview Predict Video
    for file in xd[0]:
        file_name = file[-15:-4] + '.avi'
        video_path = ORIGINAL_VIDEO_PATH + file_name
        # print(file_name)
        # print(video_path)
        try:
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow('preview', frame)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
                else:
                    break
        except cv2.error as e:
            print(e)
            False

            cap.release()


if __name__ == '__main__':
    '''
    Classify LSA 64
    '''
    # train(shuffle=True)
    # model_predict(filename="025_009_003.npz")
    '''
    '''

    '''
    CTC Full Frame
    '''
    # train_ctc()
    # train_ctc_fit(True)
    # predict_ctc()

    # demo_testing([1, 2, 3, 4, 5])
    # demo_testing([9, 59, 31, 23, 48])

    '''
    '''

    # print(y_generator(9, 10))

    # sentence_dataset_generator()

    # '''Sentence Example'''
    #
    # test_x = ['D:\\LSA64\\VGG_out_new\\063_001_003.npz', 'D:\\LSA64\\VGG_out_new\\023_001_001.npz',
    #           'D:\\LSA64\\VGG_out_new\\048_001_002.npz', 'D:\\LSA64\\VGG_out_new\\021_001_005.npz',
    #           'D:\\LSA64\\VGG_out_new\\051_001_005.npz']
    # test_y = [12, 50, 23, 48, 44]
    #
    # x, y = sentence_sequence_generator_npz(test_x, test_y)
    #
    # print(np.asarray(x).shape)
    # (700, 56, 56, 256)
    # print(y)
    # [12, 50, 23, 48, 44]

    # calculate_wer()

    x_data, y_data = generate_data_list()
    get_value(x_data)

    # train_C3D()

    # crop_dataset()

    # crop_video(r'E:\arda\LSA_64\001_007_001.mp4', '001_007_001.avi', 941, 365)

    # x_data, y_data = generate_data_list_npz()
