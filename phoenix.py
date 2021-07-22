import gc
import glob
import math
import os
import sys
from os import listdir, path
from os.path import isfile, join

import re

import natsort
import progressbar

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

from keras_ctcmodel.CTCModel import CTCModel as CTCModel

from keras.optimizers import Adam, SGD

from train_custom import TCN_layer

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

plot_losses = livelossplot.PlotLossesKeras()

import numpy as np

np.set_printoptions(threshold=sys.maxsize)

from jiwer import wer

# Config

gc.enable()

MODEL_SAVE_PATH = r'C:\Users\minelab\dev\TSL\model\Phoenix\\'

SAVED_WEIGHT_PATH = r'C:\Users\minelab\dev\TSL\model\Phoenix\model_weights.hdf5'

NPZ_PATH = r'H:\sign language\annotation\data_npz'
# NPZ_PATH = r'H:\sign language\annotation\test_npz'

LABEL_PATH = r'H:\sign language\annotation\train_alignment - Copy.txt'

# Fixed Number
FIXED_MAX = 320
FIXED_LABEL_MAX = 80
MAX_LABEL_NUMBER = 3693

STEPS = 10

FILE_START_FROM = 0  # Updated each steps !!
FILE_END = 5  # Updated each steps !!

# to continue from last frame !!
FRAME_START_FROM = 0  # Updated each steps

TOTAL_DATA_LENGTH = FILE_END - FILE_START_FROM
# TOTAL_DATA_LENGTH = STEPS

EPOCH = 10

DEBUG = True


# FRAME_END = 100

def vgg_model(i_vgg):
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

    return model


def full_model():
    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(FIXED_MAX, 56, 56, 256))

    vgg = vgg_model(i_vgg)

    '''
    TCN -> Dense
    '''
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

    '''
    Sequence Learning
    '''

    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(block2)
    dense = TimeDistributed(Dense(MAX_LABEL_NUMBER + 2, name="dense"))(blstm)  # -- 2
    outrnn = Activation('softmax', name='softmax')(dense)

    '''
    Sequence Learning # endregion
    '''

    network = CTCModel([i_vgg], [outrnn])  # -- 4
    network.compile(tf.keras.optimizers.Adam(0.00001))

    return network


# learning rate schedule
def step_decay(epoch):
    # initial_lrate = 0.00005 original LSA
    initial_lrate = 0.00001
    drop = 0.25
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def calculate_wer(gt=None, result=None, label_length=None):
    if result is None:
        result = [
            [1., 2., 3., - 1., - 1., - 1., - 1.],
            [2, 1, - 1., - 1., - 1., - 1., - 1., -1.]
        ]
    if gt is None:
        gt = [[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6]]
    if label_length is None:
        label_length = [5, 6]

    # gt2 = np.array(gt).astype(str)[:, 0:5]
    # hp2 = np.array(result).astype(int).astype(str)[:, 0:5]
    gt2 = [np.array(val).astype(int).astype(str)[0:label_length[idx]] for idx, val in enumerate(gt)]
    hp2 = [np.array(val).astype(int).astype(str)[0:label_length[idx]] for idx, val in enumerate(np.array(result))]

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

    print(gt3)
    print(hp3)

    error = wer(gt3, hp3)

    print("Word Error rate = " + str(error))


def load_data(
        npz_path=NPZ_PATH,
        label_path=LABEL_PATH,
        frame_start_from=FRAME_START_FROM,
        total_data_length=TOTAL_DATA_LENGTH,
        file_start_from=FILE_START_FROM,
        file_end=FILE_END,
        fixed_max=FIXED_MAX,
        fixed_label_max=FIXED_LABEL_MAX,
):
    print('++++ Load Data ++++')

    folders = [f.path for f in os.scandir(npz_path) if f.is_dir()]
    onlyfiles = [f for f in listdir(npz_path) if isfile(join(npz_path, f))]

    # print(folders)

    total_data = len(onlyfiles)
    print('Total Files : ', total_data)

    filePaths = sorted(glob.glob(npz_path + "/*"))
    filePaths = (natsort.os_sorted(filePaths))

    # # print(filePaths)
    # for filess in filePaths:
    #     print(filess)
    # exit(0)

    # All data
    x_data = []  # X
    y_data = []  # Y
    x_len = []  # X length
    y_len = []  # Y length

    # make list from label txt "label per frame"
    with open(label_path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    # splice label to start form particular frame / lines
    all_label = [x.split(' ')[1].strip() for x in content][frame_start_from:]

    # # debug
    # print(all_label[0])
    # exit(0)

    temp_frame_num = 0

    with progressbar.ProgressBar(max_value=total_data_length) as bar:
        for idx_file, file in enumerate(filePaths[file_start_from:file_end]):
            # print(file)

            load_ = np.load(file)
            npz_data = (load_['arr_0'])

            shape = np.asarray(npz_data).shape

            data_length = shape[0]  # frame length (supposed to be)
            data_length_dense = 77  # (dunno why it has to be changed to dense size eg: 77 for this case (?) )

            x_len.append(data_length_dense)  # append x data length

            if DEBUG:
                # print('shape : ', npz_data.shape)
                print('frame length : ', data_length)
                print('file name : ', file)

            # Handling different shape
            if len(shape) == 5:
                x_data.append(np.reshape(npz_data, (shape[0], shape[2], shape[3], shape[4])))
            else:
                x_data.append(npz_data)

            label = []

            # Read frame label
            for idx, val in enumerate(all_label[temp_frame_num:temp_frame_num + data_length]):
                # print(math.floor(int(val) / 3))
                # print(int(val) / 3)
                # print(idx)
                if idx == 0:
                    label.append((math.floor(int(val) / 3) * 3))
                elif (math.floor(int(val) / 3) * 3) == label[-1]:
                    continue
                else:
                    label.append((math.floor(int(val) / 3) * 3))

            label_length = len(label)  # label length

            y_len.append(label_length)  # append label length

            # append label
            y_data.append(label)

            if DEBUG:
                # debug
                print(label)

            temp_frame_num += data_length

            # update progressbar
            # print(idx_file)
            bar.update(idx_file)

    # print(y_data)

    print('frame count : ', temp_frame_num)
    last_frame_number = frame_start_from + temp_frame_num
    print('last frame position : ', last_frame_number)

    # exit(0)

    frame_max_length = max(x_data, key=len)
    max_frame_length = np.asarray(frame_max_length).shape[0]
    print('Max frame length :', max_frame_length)

    padded = np.empty((0, fixed_max, 56, 56, 256))
    padded_label = np.empty((0, fixed_label_max))

    with progressbar.ProgressBar(max_value=total_data_length) as bar:
        # padding data to maximum frame
        for idx, data in enumerate(x_data):
            result = np.zeros((fixed_max, 56, 56, 256))

            result[:data.shape[0], :data.shape[1], :data.shape[2], :data.shape[3]] = data

            padded = np.vstack((padded, [result]))
            bar.update(idx)

    with progressbar.ProgressBar(max_value=total_data_length) as bar:
        # padding label to maximum label length
        for idx, data in enumerate(y_data):
            result = np.zeros((fixed_label_max))

            result[:len(data)] = data

            padded_label = np.vstack((padded_label, [result]))
            bar.update(idx)

    # Debug Purpose #

    # print(np.asarray(padded).shape)
    # print(np.asarray(padded_label).shape)
    #
    # print(np.asarray(padded))
    # print(np.asarray(padded_label))
    #
    # print(x_len)
    # print(y_len)

    # X = np.asarray(padded)
    # Y = np.asarray(padded_label)
    #
    # X_length = x_len
    # Y_length = y_len

    # return x data , y data , x length ,y length

    # return X, Y, X_length, Y_length
    return padded, padded_label, x_len, y_len, last_frame_number
    # return 0, 0, 0, 0


def train_fit(
        saved_weight_path=SAVED_WEIGHT_PATH,
        epoch_per_step=EPOCH,
        npz_path=NPZ_PATH,
        label_path=LABEL_PATH,
        total_data_length=TOTAL_DATA_LENGTH,
        frame_start_from=FRAME_START_FROM,
        file_start_from=FILE_START_FROM,
        file_end=FILE_END,
):
    # Load data
    x_data, y_data, x_len, y_len, last_frame = load_data(
        file_start_from=file_start_from,
        file_end=file_end,
        frame_start_from=frame_start_from,
        total_data_length=total_data_length
    )

    gc.collect()

    print(x_data.shape)
    print(y_data.shape)
    print(y_data)
    print(x_len)
    print(y_len)

    network = full_model()

    # network.summary()

    # Load Previous Train
    prev_model_path = saved_weight_path
    if path.exists(prev_model_path):
        print('Load weight from previous training')
        network.load_weights(prev_model_path, by_name=True)

    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    y_zeros = np.zeros(total_data_length)

    network.fit(x=[x_data, y_data, np.array(x_len), np.array(y_len)],
                y=np.array(y_zeros), epochs=epoch_per_step,
                batch_size=1,
                validation_split=0.2,
                callbacks=callbacks_list)

    # temporary save to be loaded again on the next train
    network.save_model(path_dir=f'{MODEL_SAVE_PATH}')

    # checkpoint save
    directory = f'{file_start_from}-{file_end}'
    save_path = os.path.join(MODEL_SAVE_PATH, directory)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    network.save_model(path_dir=save_path)

    return network, last_frame


def train_fit_loop():
    step = 10  # aka data length, file trained per loop
    total_loop = 10
    total_trained_files = step * total_loop
    start_file = 0
    end_file = start_file + step
    frame_start = 0

    print(f'Training {total_trained_files} files')  # index
    print(f'from: {start_file} to {start_file + total_trained_files}')  # index
    print(f'frame start from: {frame_start}\n')

    for i in (range(0, total_loop)):
        print('Train step:', i)
        network, last_frame = train_fit(
            file_start_from=start_file, file_end=end_file,
            frame_start_from=frame_start, total_data_length=step,
            epoch_per_step=10
        )

        # Load data
        x_data, y_data, x_len, y_len, last_frame = load_data(
            file_start_from=0,
            file_end=3,
            frame_start_from=0,
            total_data_length=3
        )

        result = network.predict(x=[x_data, np.array(y_len)])
        words = np.where(result != -1)
        print("word", words)
        print(result)
        # print(np.asarray(result[0]).shape)

        calculate_wer(result=result, gt=y_data, label_length=y_len)

        start_file += step
        end_file = start_file + step
        frame_start = last_frame


def predict():
    network = full_model()

    network.load_model(path_dir=MODEL_SAVE_PATH, file_weights='/model_weights.hdf5',
                       optimizer=Adam(0.00001), init_last_layer=False, init_archi=False)

    network.summary()

    # Load data
    x_data, y_data, x_len, y_len, last_frame = load_data(
        file_start_from=0,
        file_end=5,
        frame_start_from=0,
        total_data_length=5
    )

    result = network.predict(x=[x_data, np.array(y_len)])
    words = np.where(result != -1)
    print("word", words)
    print(result)
    # print(np.asarray(result[0]).shape)

    calculate_wer(result=result, gt=y_data, label_length=y_len)


if __name__ == '__main__':
    print("===== Phoenix ======")

    # train_fit()

    train_fit_loop()

    # predict()

    # calculate_wer()

    # load_ = np.load(r'H:\sign language\annotation\data_npz\01October_2010_Friday_tagesschau_default-3.npz')
    # npz_data = (load_['arr_0'])
    # shape = np.asarray(npz_data).shape
    # print(shape)
