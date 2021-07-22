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
from tensorflow import keras
from tensorflow.python.keras.backend import placeholder
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

from sklearn.preprocessing import LabelBinarizer

PREVIEW = False
DEBUG = False
OUTPUT_PATH = r'D:\WLASL\wlasl_100_out_4'
MODEL_SAVE_PATH = r'D:\WLASL\model'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

MAX_FRAME = 64


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

class_count = 25

def read_dataset(filename, root):
    with open(filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    print(len(content))

    rgb_list = []
    depth_list = []
    label = []
    frame_len = []

    for text in content:
        # print(text)
        text_split = text.split()
        # print(text_split)

        if int(text_split[2]) > class_count:
            print('skip ' + text_split[0])
            continue

        rgb_list.append(text_split[0])
        depth_list.append(text_split[1])
        label.append(text_split[2])

        # Frame Length
        video_path = f'{root}\{text_split[0]}'
        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        frame_len.append(num_frames)

        # exit()

    label_list = np.unique(label)

    # print(rgb_list)
    # print(label)
    # print(label_list)
    print(len(label_list))
    print(len(content))
    print(len(rgb_list))

    return rgb_list, depth_list, label, frame_len, len(label_list)


def extract_layer_data(vids, root, output):
    with progressbar.ProgressBar(max_value=len(vids)) as bar:
        for idx, file in enumerate(vids):
            if os.path.exists(f'{root}\{file[:-4]}.npz'):
                # if DEBUG:
                #     print(f'file {file}.npz already exist')
                pass
            else:
                extract_video(f'{root}\{file}', file, output)
            bar.update(idx)


def extract_video(file, file_path, output_root):
    if DEBUG:
        print(file)

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

        if not os.path.exists(f'{output_root}\{file_path[:-12]}'):
            os.makedirs(f'{output_root}\{file_path[:-12]}')

        savez_compressed(f'{output_root}\{file_path[:-4]}.npz', output)

        # exit(0)
        cap.release()

    except cv2.error as e:
        print(e)
        False

    tf.keras.backend.clear_session()
    # print('save npz')


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


def TCN_layer(input_layer, kernel):
    #    inputs=Input(shape=(28,28))
    # print(input_layer)
    x = ResBlock(input_layer, filters=64, kernel_size=kernel, dilation_rate=1)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=2)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=4)
    x = ResBlock(x, filters=64, kernel_size=kernel, dilation_rate=8)
    #    x=Flatten()(x)
    return x


def split_list(a_list, factor=0.5):
    length = len(a_list)
    split1 = int(len(a_list) * factor)
    split2 = length - split1
    return a_list[:split2], a_list[split1:]


def train(x_data_, y_data, frame_len, root, shuffle=True, num_classes=0):
    max_frame_len = 96

    # override max_len
    max_len = max_frame_len

    # # shuffle data
    # if shuffle:
    #     c = list(zip(x_data_, y_data, frame_len))
    #     random.shuffle(c)
    #     x_data_, y_data, frame_len = zip(*c)

    x_data = []

    # convert x to video path
    for file in x_data_:
        x_data.append(f'{root}\{file[:-4]}.npz')

    # convert label to 1 hot encoder
    lb = LabelBinarizer()
    y_data = lb.fit_transform(y_data)
    # y_data = keras.utils.to_categorical(y_data)

    x_data, x_data_validate = split_list(x_data, 0.1)
    y_data, y_data_validate = split_list(y_data, 0.1)
    frame_len, frame_len_validate = split_list(frame_len, 0.1)

    print(x_data[0])
    print(y_data[0])

    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(max_len, 56, 56, 256))

    vgg = VGG(i_vgg)
    m_vgg = Model(inputs=[i_vgg], outputs=[vgg])

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

    # # blstm,
    # blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(block2)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)

    # classification
    flatten = Flatten()(block2)  # using flatten to sync the network size
    dense = Dense(256, activation='relu')(flatten)
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
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=0.0001, epsilon=1e-3, amsgrad=False,
    #     name='Adam'
    # )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)

    # Compile
    metrics_names = ['acc']
    tcn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics_names)

    # # Load Previous Train
    # if os.path.exists(f'{MODEL_SAVE_PATH}\TMC_Classification_Isolated_25.h5'):
    #     # network.load_weights(r'C:\Users\minelab\dev\TSL\model\LSA64\model_weights.hdf5', by_name=True)
    #     print('Load previous model')
    #     tcn_model = load_model(f'{MODEL_SAVE_PATH}\TMC_Classification_Isolated_25.h5')
    #     print('model loaded.')
    #     tcn_model.summary()

    batch_size = 4
    epochs = 10

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

            x_list = []
            y_list = []

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                y_list.append(y[i_data])

                v_length = vid_length[i_data]
                start_f = random.randint(0, v_length - 1)

                if start_f > (v_length - (max_frame_len + 1)):
                    start_f = 0

                x_npz = np.load(X[i_data])
                x_list.append(np.asarray(x_npz['arr_0'][start_f:start_f + max_frame_len, :, :, :]))

            x_list = tf.keras.preprocessing.sequence.pad_sequences(
                x_list, padding="post", maxlen=max_len,
            )

            # current_learning_rate = step_decay(epoch)
            # tf.keras.backend.set_value(tcn_model.optimizer.learning_rate, current_learning_rate)

            # print(np.array(x_list).shape)
            # print(np.array(y_list).shape)

            X = tcn_model.train_on_batch(np.array(x_list), np.array(y_list))

            values = [('acc', X[1])]
            pb_i.add(batch_size, values=values)

            acc.append(X[1])
            loss.append(X[0])

        # Validate dataset
        for i in range(0, len(x_data_validate) // batch_size):
            X_val = x_data_validate[i * batch_size:min(len(x_data_validate), (i + 1) * batch_size)]
            y_val = y_data_validate[i * batch_size:min(len(y_data_validate), (i + 1) * batch_size)]
            vid_length = frame_len_validate[i * batch_size:min(len(frame_len_validate), (i + 1) * batch_size)]

            x_list_val = []
            y_list_val = []

            for i_data in range(0, len(X_val)):
                y_list_val.append(y_val[i_data])

                v_length = vid_length[i_data]
                start_f = random.randint(0, v_length - 1)

                if start_f > (v_length - (max_frame_len + 1)):
                    start_f = 0

                x_npz = np.load(X_val[i_data])
                x_list_val.append(np.asarray(x_npz['arr_0'][start_f:start_f + max_frame_len, :, :, :]))

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
                filepath=f'{MODEL_SAVE_PATH}\{"TMC_Classification_Isolated"}_{num_classes}_{loss_}.h5')

    tcn_model.save(
        filepath=f'{MODEL_SAVE_PATH}\{"TMC_Classification_Isolated"}_{num_classes}.h5')

    print(acc_model)
    print(loss_model)

    print(acc_model_val)
    print(loss_model_val)

    print('train done.')


if __name__ == '__main__':
    rgb_list, depth_list, label, frame_len, total_class = read_dataset(r'D:\Isolated\IsoGD_labels\train.txt',
                                                          r'D:\Isolated\train (1)')

    # extract_layer_data(rgb_list, r'D:\Isolated\train (1)', r'F:\Dataset\Sign Language\Isolated_Chalearn16\Extract')

    train(rgb_list, label, frame_len, num_classes=total_class, root=r'F:\Dataset\Sign Language\Isolated_Chalearn16\Extract')
    # lb = LabelBinarizer()
    # label = lb.fit_transform(label)
    #
    # print(label)
