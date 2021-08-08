import math
import os
import json
import random
import zipfile

import cv2
import numpy as np
from keras.optimizer_v2.adam import Adam
from numpy import savez_compressed, savez
import tensorflow as tf
# import tensorflow.keras as keras
import progressbar
from tensorflow.python.keras.backend import placeholder, concatenate
from tensorflow.python.keras.models import load_model

from extract_layer4 import get_output_layer
import gc
from tensorflow.python.keras.layers import Conv1D, Add, Activation, Lambda, Dense, TimeDistributed, Conv2D, \
    MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM, Dropout, MaxPooling1D, Bidirectional, MaxPool2D, UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar

from train_custom import ResBlock

from sklearn import preprocessing

import pandas as pd

PREVIEW = False
DEBUG = False
OUTPUT_PATH = r'D:\WLASL\wlasl_100_out_4'

OUTPUT_PATH = r'F:\Dataset\Sign Language\WLASL-Alter\Output'

MODEL_SAVE_PATH = r'F:\WLASL\model2'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

MAX_FRAME = 64

KEYPOINT_PATH = r"F:\Dataset\Sign Language\WLASL-Alter\Key"


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.25
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)


test_class = [0, 1]
get_class = [0, 1, 2, 3, 4]

skip_list = [
    "07073",
    "17728", "17722", "17721", "17718", "65540", "17733",
    "68028", "12306", "12316", "12317", "12314", "12312", "12318", "12319", "12323", "12322", "12321", "12327", "12326",
    "12335", "12331", "12332", "69054",
    "05736", "05730", "05733", "05739", "05740", "05744", "05747", '05750', "65167", "05748", "05729", "05727",
    "09863",

    "05743", "70266"
]

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


def make_dataset(split_file, split, root, mode, num_classes):
    dataset = []
    label = []
    frame_len = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    count_skipping = 0
    for vid in data.keys():
        if split == 'train':
            # if data[vid]['subset'] not in ['train', 'val']:
            if data[vid]['subset'] not in ['train']:
                continue
        elif split == 'val':
            if data[vid]['subset'] not in ['val']:
                continue
        elif split == 'all':
            pass
        else:
            if data[vid]['subset'] != 'test':
                continue

        # root directory
        vid_root = root['word']
        src = 0

        video_path = os.path.join(vid_root, vid + '.mp4')
        if not os.path.exists(video_path):
            continue

        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        # # if (data[vid]['action'][2] - data[vid]['action'][1]) > MAX_FRAME:
        # if num_frames > MAX_FRAME:
        #     print("Skip video ", vid)
        #     count_skipping += 1
        #     continue

        if vid in skip_list:
            print("Skip video ", vid)
            count_skipping += 1
            continue

        if data[vid]['action'][0] not in get_class:
            print("Skip video ", vid)
            count_skipping += 1
            continue

        # if data[vid]['action'][0] not in test_class:
        #     print("Skip video ", vid)
        #     count_skipping += 1
        #     continue

        frame_len.append(num_frames)

        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames - 0 < 9:
            print("Skip video ", vid)
            count_skipping += 1
            continue

        if mode == 'filename':
            i += 1
            dataset.append(vid)
            label.append(data[vid]['action'][0])
            continue

        label = np.zeros((num_classes, num_frames), np.float32)

        for l in range(num_frames):
            c_ = data[vid]['action'][0]
            label[c_][l] = 1

        print(label.shape)
        print(num_frames)
        # print(label[data[vid]['action'][0]])
        print(data[vid]['action'][0])
        # split 100 goes here
        if len(vid) == 5:
            dataset.append((vid, label, src, 0, data[vid]['action'][2] - data[vid]['action'][1]))
            # print(data[vid]['action'][2])
            # print(data[vid]['action'][1])
            # print(data[vid]['action'][2] - data[vid]['action'][1])
        elif len(vid) == 6:  ## sign kws instances
            dataset.append((vid, label, src, data[vid]['action'][1], data[vid]['action'][2] - data[vid]['action'][1]))

        i += 1
    print("Skipped videos: ", count_skipping)
    print(len(dataset))
    return dataset, label, np.max(frame_len), frame_len


dataset_path = r'E:\WLASL2000'


def extract_layer_data(vids):
    with progressbar.ProgressBar(max_value=len(vids)) as bar:
        for idx, file in enumerate(vids):
            if os.path.exists(f'{OUTPUT_PATH}\{file}.npz'):
                # if DEBUG:
                #     print(f'file {file}.npz already exist')
                pass
            else:
                extract_video(f'{dataset_path}\{file}.mp4', file)
            bar.update(idx)


def extract_video(file, fileName):
    if DEBUG:
        print(file)
        print(fileName)

    try:
        cap = cv2.VideoCapture(file)
        video = []

        while cap.isOpened():
            ret, frame = cap.read()
            # (height, width) = frame.shape[:2]
            if ret:

                if PREVIEW:
                    cv2.imshow('orig', frame)

                resized_image = cv2.resize(frame, (224, 224))

                # normalize
                resized_image = (resized_image / 255.) * 2 - 1

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

        # savez(f'{OUTPUT_PATH}\{fileName}.npz', output)
        savez_compressed(f'{OUTPUT_PATH}\{fileName}.npz', output)

        # exit(0)

        cap.release()

    except cv2.error as e:
        print(e)
        False

    tf.keras.backend.clear_session()

    # print('save npz')


def crop_video(file):
    try:
        cap = cv2.VideoCapture(file)
        video = []

        while cap.isOpened():
            ret, frame = cap.read()
            # (height, width) = frame.shape[:2]
            if ret:

                if PREVIEW:
                    cv2.imshow('orig', frame)

                resized_image = cv2.resize(frame, (96, 96))

                if PREVIEW:
                    cv2.imshow('resized', resized_image)

                # append frame to be converted
                video.append(np.asarray(resized_image))

                if PREVIEW:
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            else:
                break

        return video

    except cv2.error as e:
        print(e)
        False


def get_frame(file):
    try:
        cap = cv2.VideoCapture(file)
        video = []

        mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2

        cap.set(1, mid)

        ret, frame = cap.read()

        resized_image = None

        if ret:
            if PREVIEW:
                cv2.imshow('orig', frame)
            resized_image = cv2.resize(frame, (224, 224))

            # normalize
            resized_image = (resized_image / 255.) * 2 - 1

            if PREVIEW:
                cv2.imshow('resized', resized_image)

        return resized_image

    except cv2.error as e:
        print(e)
        False


def VGG(i_vgg):
    model1 = TimeDistributed(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", name='block3_conv3'))(i_vgg)
    model1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name='block4_conv1'))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name='block4_conv2'))(model1)

    model1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model1)
    model = TimeDistributed(GlobalAveragePooling2D(name="global_max_full"))(model1)

    #    model = (MaxPooling3D(pool_size=(1, 2, 2)))(model)
    #    model1 = (TimeDistributed(GlobalAveragePooling2D(name="global_max_full")))(model1)

    #    model.compile(loss='mean_squared_error', optimizer='adam')  #,

    return model


def VGGfull(i_vgg):
    model = TimeDistributed(
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name='block1_conv1'))(i_vgg)
    model = TimeDistributed(
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name='block1_conv2'))(model)
    model = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model)
    model = TimeDistributed(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", name='block2_conv1'))(model)
    model = TimeDistributed(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", name='block2_conv2'))(model)
    model = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model)
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

    model1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model1)
    model = TimeDistributed(GlobalAveragePooling2D(name="global_max_full"))(model1)

    #    model = (MaxPooling3D(pool_size=(1, 2, 2)))(model)
    #    model1 = (TimeDistributed(GlobalAveragePooling2D(name="global_max_full")))(model1)

    #    model.compile(loss='mean_squared_error', optimizer='adam')  #,

    return model


def TCN_layer(input_layer, kernel):
    #    inputs=Input(shape=(28,28))
    # print(input_layer)
    x = ResBlock(input_layer, filters=64, kernel_size=kernel, dilation_rate=1)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=2)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=4)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=8)
    #    x=Flatten()(x)
    return x


def train(split_file, split, root, mode, num_classes=0, shuffle=True):
    max_frame_len = 96

    x_data = []
    x_data_keypoint = []
    x_data_validate = []
    x_data_validate_keypoint = []
    x_data_, y_data, max_len, frame_len = make_dataset(split_file, split, root, mode, num_classes=num_classes)
    x_data_validate_, y_data_validate, max_len_validate, frame_len_validate = make_dataset(split_file, 'val', root,
                                                                                           mode,
                                                                                           num_classes=num_classes)

    print(x_data_)
    print(len(x_data_))
    print(x_data_validate_)
    print(len(x_data_validate_))

    print(max_len)
    print(max_len_validate)

    print(frame_len)
    print(max_len_validate)

    # override max_len
    max_len = max_frame_len

    # shuffle data
    if shuffle:
        c = list(zip(x_data_, y_data, frame_len))
        random.shuffle(c)
        x_data_, y_data, frame_len = zip(*c)

    # convert label to 1 hot encoder
    y_data = tf.keras.utils.to_categorical(y_data, )
    y_data_validate = tf.keras.utils.to_categorical(y_data_validate)

    # convert x to video path
    for file in x_data_:
        x_data.append(f'{OUTPUT_PATH}\{file}.npz')
        x_data_keypoint.append(f'{KEYPOINT_PATH}\{file}.mp4.npy')

    # convert x to video path
    for file in x_data_validate_:
        x_data_validate.append(f'{OUTPUT_PATH}\{file}.npz')
        x_data_validate_keypoint.append(f'{KEYPOINT_PATH}\{file}.mp4.npy')

    print(x_data[0])
    print(y_data[0])

    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(max_len, 56, 56, 256))

    i_keypoint = tf.keras.Input(name="input_1_keypoint", shape=(max_len, 1, 27, 3))
    output_keypoint = TimeDistributed(GlobalAveragePooling2D(name="global_max_full"))(i_keypoint)
    dense_input_keypoint = Dense(256, activation='relu', name='dense_keypoint')(output_keypoint)

    vgg = VGG(i_vgg)
    m_vgg = Model(inputs=[i_vgg], outputs=[vgg])

    # TCN

    '''
    TCN -> Dense
    '''

    o_tcn_full = TCN_layer(vgg, 5)
    o_tcn_keypoint = TCN_layer(dense_input_keypoint, 5)
    # global_pool = GlobalAveragePooling2D(name="global_max_full")(o_tcn_full)
    # flatten = Flatten()(o_tcn_full) # using flatten to sync the network size - disabled if 'TMC FULL'
    dense = Dense(256, activation='relu', name='dense_o_tcn1')(o_tcn_full)
    dense_keypoint = Dense(256, activation='relu', name='dense_o_keypoint')(o_tcn_keypoint)

    '''
    TMC (cont)
    '''

    o_tcn_block1 = TCN_layer(dense, 1)
    o_tcn_block1 = Dense(256, name='dense_o_tcn_intra_block1')(o_tcn_block1)
    o_tcn_block1 = Dense(256)(o_tcn_block1)
    block1 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block1)

    o_tcn_key_block1 = TCN_layer(dense_keypoint, 1)
    o_tcn_key_block1 = Dense(256, name='dense_o_tcn_key_intra_block1')(o_tcn_key_block1)
    o_tcn_key_block1 = Dense(256)(o_tcn_key_block1)
    block1_key = MaxPooling1D(pool_size=5, strides=2)(o_tcn_key_block1)

    i_tcn2 = block1
    o_tcn2 = TCN_layer(i_tcn2, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2 = Dense(256, activation='relu', name='dense_o_tcn2')(o_tcn2)
    o_tcn_block2 = TCN_layer(dense2, 1)
    o_tcn_block2 = Dense(256, name='dense_o_tcn_intra_block2')(o_tcn_block2)
    o_tcn_block2 = Dense(256)(o_tcn_block2)
    block2 = MaxPooling1D(pool_size=5, strides=2)(o_tcn_block2)

    i_tcn2_key = block1_key
    o_tcn2_key = TCN_layer(i_tcn2_key, 5)
    # flatten2 = Flatten()(o_tcn2) # using flatten to sync the network size
    dense2_key = Dense(256, activation='relu', name='dense_o_tcn2_key')(o_tcn2_key)
    o_tcn_key_block2 = TCN_layer(dense2_key, 1)
    o_tcn_key_block2 = Dense(256, name='dense_o_tcn_key_intra_block2')(o_tcn_key_block2)
    o_tcn_key_block2 = Dense(256)(o_tcn_key_block2)
    block2_key = MaxPooling1D(pool_size=5, strides=2)(o_tcn_key_block2)

    '''
    TMC (cont) # endregion
    '''

    # # blstm,
    # blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(block2)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)

    concat = concatenate([block2, block2_key], axis=2)

    # classification
    flatten = Flatten()(concat)  # using flatten to sync the network size
    dense = Dense(256, activation='relu')(flatten)
    # endregion

    # classification total class - dense
    dense = Dense(num_classes, activation='softmax')(dense)
    tcn_model = Model(inputs=[i_vgg, i_keypoint], outputs=[dense])

    tcn_model.summary()
    # exit(0)

    # # Full Frame Model

    ### The Custom Loop
    # The train_on_batch function
    loss = tf.keras.losses.categorical_crossentropy
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=0.0001, epsilon=1e-3, amsgrad=False,
    #     name='Adam'
    # )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)

    # Compile
    metrics_names = ['acc']
    tcn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics_names)

    # # Load Previous Train
    # if os.path.exists(f'{MODEL_SAVE_PATH}\TMC_Classification_WLASL_5.h5'):
    #     # network.load_weights(r'C:\Users\minelab\dev\TSL\model\LSA64\model_weights.hdf5', by_name=True)
    #     print('Load previous model')
    #     tcn_model = load_model(f'{MODEL_SAVE_PATH}\TMC_Classification_WLASL_5.h5')
    #     print('model loaded.')
    #     tcn_model.summary()

    batch_size = 4
    epochs = 100

    loss_ = 999999999

    acc_model = []
    loss_model = []

    acc_model_val = []
    loss_model_val = []

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch}')

        # shuffle data again
        if shuffle:
            c = list(zip(x_data, y_data, frame_len))
            random.shuffle(c)
            x_data, y_data, frame_len = zip(*c)

        acc = []
        loss = []
        val_acc = []
        val_loss = []

        pb_i = Progbar(len(x_data), width=60, stateful_metrics=['acc'])

        for i in range(0, len(x_data) // batch_size):
            X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
            y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]
            vid_length = frame_len[i * batch_size:min(len(frame_len), (i + 1) * batch_size)]
            X_keypoint = x_data_keypoint[i * batch_size:min(len(x_data_keypoint), (i + 1) * batch_size)]

            x_list = []
            y_list = []
            x_key_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                y_list.append(y[i_data])

                v_length = vid_length[i_data]
                start_f = random.randint(0, v_length - 1)

                if start_f > (v_length - (max_frame_len + 1)):
                    start_f = 0

                x_npz = np.load(X[i_data])
                x_list.append(np.asarray(x_npz['arr_0'][start_f:start_f + max_frame_len, :, :, :]))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]
                # print(x_npy.shape)
                x_npy = x_npy.reshape((x_npy.shape[0],1,27,3))
                x_key_list.append(np.asarray(x_npy[start_f:start_f + max_frame_len, :, :, :]))

            x_list = tf.keras.preprocessing.sequence.pad_sequences(
                x_list, padding="post", maxlen=max_len,
            )

            x_key_list = tf.keras.preprocessing.sequence.pad_sequences(
                x_key_list, padding="post", maxlen=max_len,
            )

            # current_learning_rate = step_decay(epoch)
            # tf.keras.backend.set_value(tcn_model.optimizer.learning_rate, current_learning_rate)

            # print(np.array(x_list).shape)
            # print(np.array(y_list).shape)

            X = tcn_model.train_on_batch([np.array(x_list), np.array(x_key_list)], np.array(y_list))

            values = [('acc', X[1])]
            pb_i.add(batch_size, values=values)

            acc.append(X[1])
            loss.append(X[0])

        # Validate dataset
        for i in range(0, len(x_data_validate) // batch_size):
            X_val = x_data_validate[i * batch_size:min(len(x_data_validate), (i + 1) * batch_size)]
            y_val = y_data_validate[i * batch_size:min(len(y_data_validate), (i + 1) * batch_size)]
            vid_length = frame_len_validate[i * batch_size:min(len(frame_len_validate), (i + 1) * batch_size)]

            X_keypoint_val = x_data_validate_keypoint[i * batch_size:min(len(x_data_validate_keypoint), (i + 1) * batch_size)]

            x_list_val = []
            y_list_val = []
            x_key_list_val = []

            for i_data in range(0, len(X_val)):
                y_list_val.append(y_val[i_data])

                v_length = vid_length[i_data]
                start_f = random.randint(0, v_length - 1)

                if start_f > (v_length - (max_frame_len + 1)):
                    start_f = 0

                x_npz = np.load(X_val[i_data])
                x_list_val.append(np.asarray(x_npz['arr_0'][start_f:start_f + max_frame_len, :, :, :]))

                x_npy = np.load(X_keypoint_val[i_data])
                x_npy = x_npy[:, selected, :]
                # print(x_npy.shape)
                x_npy = x_npy.reshape((x_npy.shape[0],1,27,3))
                x_key_list_val.append(np.asarray(x_npy[start_f:start_f + max_frame_len, :, :, :]))

            x_list_val = tf.keras.preprocessing.sequence.pad_sequences(
                x_list_val, padding="post", maxlen=max_len,
            )

            x_key_list_val = tf.keras.preprocessing.sequence.pad_sequences(
                x_key_list_val, padding="post", maxlen=max_len,
            )

            X_val = tcn_model.test_on_batch([np.array(x_list_val), np.array(x_key_list_val)], np.array(y_list_val))

            val_acc.append(X_val[1])
            val_loss.append(X_val[0])

        print(f'Loss : {np.average(np.array(loss))}, Accuracy : {np.average(np.array(acc))}')
        print(f'Val Loss : {np.average(np.array(val_loss))}, Val Accuracy : {np.average(np.array(val_acc))}')

        acc_model.append(np.average(np.array(acc)))
        loss_model.append(np.average(np.array(loss)))

        acc_model_val.append(np.average(np.array(val_acc)))
        loss_model_val.append(np.average(np.array(val_loss)))

        if np.average(np.array(loss)) < loss_:
            loss_ = np.average(np.array(loss))
            tcn_model.save(
                filepath=f'{MODEL_SAVE_PATH}\{"TMC_Classification_WLASL"}_{num_classes}_{loss_}.h5')

    tcn_model.save(
        filepath=f'{MODEL_SAVE_PATH}\{"TMC_Classification_WLASL"}_{num_classes}.h5')

    print(acc_model)
    print(loss_model)

    print(acc_model_val)
    print(loss_model_val)

    print('train done.')


# VIDEO_PATH = r"C:\Users\minelab\dev\TSL\WLASL\dataset"
VIDEO_PATH = r"E:\WLASL2000"


def train_video(split_file, split, root, mode, num_classes=0, shuffle=False):
    x_data = []
    x_data_validate = []
    x_data_, y_data, max_len = make_dataset(split_file, split, root, mode, num_classes=num_classes)
    x_data_validate_, y_data_validate, max_len_validate = make_dataset(split_file, 'val', root, mode,
                                                                       num_classes=num_classes)

    # shuffle data
    if shuffle:
        c = list(zip(x_data_, y_data))
        random.shuffle(c)
        x_data_, y_data = zip(*c)

    # convert label to 1 hot encoder
    y_data = tf.keras.utils.to_categorical(y_data)
    y_data_validate = tf.keras.utils.to_categorical(y_data_validate)
    # convert x to video path
    for file in x_data_:
        x_data.append(f'{VIDEO_PATH}\{file}.mp4')

    # convert x to video path
    for file in x_data_validate_:
        x_data_validate.append(f'{VIDEO_PATH}\{file}.mp4')

    print(x_data[0])
    print(y_data[0])

    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)

    # Load Previous Train
    if os.path.exists(f'{MODEL_SAVE_PATH}\TMC_VIDEO_WLASL_100.h5'):
        # network.load_weights(r'C:\Users\minelab\dev\TSL\model\LSA64\model_weights.hdf5', by_name=True)
        print('Load previous model')
        tcn_model = load_model(f'{MODEL_SAVE_PATH}\TMC_VIDEO_WLASL_100.h5')
        print('model loaded.')
        metrics_names = ['acc']
    else:
        # Input from intermediate layer
        i_vgg = tf.keras.Input(name="input_1", shape=(max_len, 96, 96, 3))

        vgg = VGGfull(i_vgg)

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
        dense = Dense(128, activation='relu')(dense)
        # endregion

        # classification total class - dense
        dense = Dense(num_classes, activation='softmax')(dense)
        tcn_model = Model(inputs=[i_vgg], outputs=[dense])

        tcn_model.summary()
        # exit(0)

        # # Full Frame Model

        ### The Custom Loop
        # The train_on_batch function
        loss = tf.keras.losses.categorical_crossentropy
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, epsilon=1e-3, amsgrad=False,
            name='Adam'
        )
        # optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)

        # Compile
        metrics_names = ['acc']
        tcn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics_names)

    batch_size = 1
    epochs = 5

    loss_ = 999999999

    acc_model = []
    loss_model = []

    acc_model_val = []
    loss_model_val = []

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch}')

        acc = []
        loss = []
        val_acc = []
        val_loss = []

        pb_i = Progbar(len(x_data), width=50, stateful_metrics=metrics_names)

        for i in range(0, len(x_data) // batch_size):
            X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
            y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]

            x_list = []
            y_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                x_list.append(NormalizeData(crop_video(X[i_data])))
                y_list.append(y[i_data])

            x_list = tf.keras.preprocessing.sequence.pad_sequences(
                x_list, padding="post", maxlen=max_len,
            )

            X = tcn_model.train_on_batch(np.array(x_list), np.array(y_list))

            values = [('acc', X[1])]

            pb_i.add(batch_size, values=values)

            acc.append(X[1])
            loss.append(X[0])

        # Validate dataset
        for i in range(0, len(x_data_validate) // batch_size):
            X_val = x_data_validate[i * batch_size:min(len(x_data_validate), (i + 1) * batch_size)]
            y_val = y_data_validate[i * batch_size:min(len(y_data_validate), (i + 1) * batch_size)]

            x_list_val = []
            y_list_val = []

            for i_data in range(0, len(X_val)):
                x_list_val.append(NormalizeData(crop_video(X_val[i_data])))
                y_list_val.append(y_val[i_data])

            x_list_val = tf.keras.preprocessing.sequence.pad_sequences(
                x_list_val, padding="post", maxlen=max_len,
            )

            X_val = tcn_model.test_on_batch(np.array(x_list_val), np.array(y_list_val))

            val_acc.append(X_val[1])
            val_loss.append(X_val[0])

        print(f'Loss : {np.average(np.array(loss))}, Accuracy : {np.average(np.array(acc))}')
        print(f'Val Loss : {np.average(np.array(val_loss))}, Val Accuracy : {np.average(np.array(val_acc))}')

        acc_model.append(np.average(np.array(acc)))
        loss_model.append(np.average(np.array(loss)))

        acc_model_val.append(np.average(np.array(val_acc)))
        loss_model_val.append(np.average(np.array(val_loss)))

        if np.average(np.array(loss)) < loss_:
            loss_ = np.average(np.array(loss))
            tcn_model.save(
                filepath=f'{MODEL_SAVE_PATH}\{"TMC_VIDEO_WLASL"}_{num_classes}_{loss_}.h5')

    tcn_model.save(
        filepath=f'{MODEL_SAVE_PATH}\{"TMC_VIDEO_WLASL"}_{num_classes}.h5')

    print(acc_model)
    print(loss_model)

    print('train done.')


def train_weight(split_file, split, root, mode, num_classes=0, shuffle=True):
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # config = tf.ConfigProto(device_count={'GPU': 0})
    inputs = tf.keras.Input(name="input_1", shape=(224, 224, 3))

    x = Conv2D(name="block1_conv1", input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3),
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
    x = MaxPool2D(name="block3_pool", pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(name="block4_conv1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(name="block4_conv2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(name="block4_conv3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(name="block4_pool", pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(name="block5_conv1", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(name="block5_conv2", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(name="block5_conv3", filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(name="block5_pool", pool_size=(2, 2), strides=(2, 2))(x)

    # # Pixel Map
    # x = Conv2D(filters=CHANNEL, kernel_size=(3, 3), padding="same", activation="relu")(x)
    #
    x = UpSampling2D(size=(2, 2,))(x)
    # x = UpSampling2D(size=(2, 2,))(x)
    # Conv2D(CHANNEL, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

    # Keypoint
    x = Flatten()(x)
    # x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    # model.compile(optimizer="Adam", loss="mse", metrics=["mae", "accuracy"])

    # opt = Adam(lr=0.0001)

    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001, epsilon=1e-3, amsgrad=False,
        name='Adam'
    )
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    metrics_names = ['acc']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_names)
    # model.compile(optimizer='sgd', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy', 'mae'])

    model.summary()

    x_data = []
    x_data_, y_data, max_len = make_dataset(split_file, split, root, mode, num_classes=num_classes)

    # shuffle data
    if shuffle:
        c = list(zip(x_data_, y_data))
        random.shuffle(c)
        x_data_, y_data = zip(*c)

    # convert label to 1 hot encoder
    y_data = tf.keras.utils.to_categorical(y_data)
    # convert x to video path
    for file in x_data_:
        x_data.append(f'{VIDEO_PATH}\{file}.mp4')

    # Load Previous Train
    if os.path.exists(f'{MODEL_SAVE_PATH}\TMC_VIDEO_WLASL_100.h5'):
        # network.load_weights(r'C:\Users\minelab\dev\TSL\model\LSA64\model_weights.hdf5', by_name=True)
        print('Load previous model')
        tcn_model = load_model(f'{MODEL_SAVE_PATH}\TMC_VIDEO_WLASL_100.h5')
        print('model loaded.')

    batch_size = 256
    epochs = 25

    loss_ = 999999999

    acc_model = []
    loss_model = []

    for epoch in range(0, epochs):
        print(f'EPOCH : {epoch}')

        acc = []
        loss = []

        pb_i = Progbar(len(x_data), stateful_metrics=metrics_names)

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

                x_list.append(get_frame(X[i_data]))
                y_list.append(y[i_data])

            # # print(x_list)
            # print(np.asarray(x_list).shape)
            # print(np.asarray(y_list).shape)

            current_learning_rate = step_decay(epoch)
            tf.keras.backend.set_value(model.optimizer.learning_rate, current_learning_rate)

            X = model.train_on_batch(np.array(x_list), np.array(y_list))

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
            model.save(
                filepath=f'{MODEL_SAVE_PATH}\{"TMC_VIDEO_WLASL"}_{num_classes}_{loss_}.h5')

        # val_loss = []
        # for i in range(0, len(x_data) // batch_size):
        #     X = x_data[i * batch_size:min(len(x_data), (i + 1) * batch_size)]
        #     y = y_data[i * batch_size:min(len(y_data), (i + 1) * batch_size)]
        #     val_loss.append(tcn_model.validate_on_batch(X, y))
        #
        # print('Validation Loss: ' + str(np.mean(val_loss)))

    model.save(
        filepath=f'{MODEL_SAVE_PATH}\{"TMC_VIDEO_WLASL"}_{num_classes}.h5')

    print(acc_model)
    print(loss_model)

    print('train weight done.')


def verify_npz(files):
    x_data = []
    for file in files:
        x_data.append(f'{OUTPUT_PATH}\{file}.npz')

    data1 = None

    for file in x_data:
        print(file)
        x_npz = np.load(file, mmap_mode='r')

        print(x_npz['arr_0'])

        gc.collect()


if __name__ == '__main__':
    # WLASL setting
    mode = 'rgb'
    root = {'word': r'C:\Users\minelab\dev\TSL\WLASL\dataset'}

    save_model = 'checkpoints/'
    split_file = r'C:\Users\minelab\dev\TSL\WLASL\code\I3D\preprocess\nslt_100.json'

    num_classes = get_num_class(split_file)
    print(f'Training {num_classes} classes')
    # data = make_dataset(split_file, 'all', root, mode, num_classes=num_classes)

    # 5 class
    # root = {'word': r'F:\Dataset\Sign Language\WLASL-Alter\Video'}
    # data, label, length, x = make_dataset(split_file, 'all', root, 'filename', num_classes=num_classes)
    #
    # print(data)
    # print(len(data))
    # extract_layer_data(data)
    # exit()

    '''
    # Extract 4th Layer
    # '''
    # data, label, length = make_dataset(split_file, 'all', root, 'filename', num_classes=num_classes)
    # extract_layer_data(data)
    # exit()
    '''
    Verify NPZ
    '''
    # data, label, length = make_dataset(split_file, 'all', root, 'filename', num_classes=num_classes)
    # data = ['32162']
    # verify_npz(data)
    '''
    Training
    '''
    num_classes = len(get_class)

    split = 'train'
    # train_weight(split_file, split, root, 'filename', num_classes=num_classes)
    # root = {'word': r'F:\Dataset\Sign Language\WLASL-Alter\Output'}
    train(split_file, split, root, 'filename', num_classes=num_classes)
    # train_video(split_file, split, root, 'filename', num_classes=num_classes)
    #

    # one_hot_encode = keras.utils.to_categorical(label)

    # x_npz = np.load(r"D:\WLASL\wlasl_100_out_4\10892.npz")
    #
    # x_npz_crop = x_npz['arr_0'][0:10, :, :, :]
    #
    # print(np.asarray(x_npz['arr_0']).shape)
    # print(np.asarray(x_npz_crop).shape)
