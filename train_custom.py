import glob
import os
from os import listdir
from os.path import isfile, join

import gc
import tensorflow as tf
from keras.preprocessing import sequence
from tensorflow import keras

from tensorflow.keras.models import Model
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Add, Activation, Lambda, Dense, TimeDistributed, Conv2D, \
    MaxPooling2D, GlobalAveragePooling2D

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from PIL import Image
from numpy import savez_compressed
import numpy as np
import cv2
import random

MODEL = r"C:\Users\minelab\dev\TSL\model\model_3_point_full_data.h5"

DIR = r'D:\TSL\OUT_Layer_4'

TOTAL_CLASS = 10


def generate_data_list():
    print("Generate X")
    folders = [f.path for f in os.scandir(DIR) if f.is_dir()]

    print(folders)

    all_files = []

    count = 0

    for folder in folders:
        # print(folder)
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        # print(files)

        filePaths = glob.glob(folder + "/*")

        if count == TOTAL_CLASS:
            break
        count += 1

        for file in filePaths:
            # print(file)
            all_files.append(file)

    print(np.asarray(all_files).shape)
    # exit(0)

    return all_files


output_data_path = r'D:\TSL\output_sentence.txt'
total_sentence_groundtruth = TOTAL_CLASS
target_unique_text = set()
list_num_samples_per_sentence = 250


def generate_y(target_unique_text=target_unique_text):
    print("Generate Y")
    """
    3. Manage Output Sentence Data //3.1. get the unique words index (dictionary)
    """
    with open(output_data_path, 'r', encoding='utf-8') as f:
        # class
        lines = f.read().split('\n')
        index_sentence = 0
        target_max_length = 0
        for line in lines[: min(total_sentence_groundtruth, len(lines) - 1)]:  # get X rows of training instances
            print('line: ', line)
            # print('line2',min(total_sentence_groundtruth, len(lines)))
            target_text = line.split('（')[1][:-1]
            # print("target_text", target_text)                           #for 1 sentence

            array_target_text = target_text.split(' ')  # change to array to find max length of words
            if len(array_target_text) > target_max_length:
                target_max_length = len(array_target_text)

            for word in array_target_text:
                if word not in target_unique_text and word != ' ':  # not count the space
                    target_unique_text.add(word)

            index_sentence += 1

    target_unique_text = sorted(list(target_unique_text))
    nb_labels = len(target_unique_text)
    # print('Number of unique output tokens:', nb_labels)
    # print('Max sequence length for outputs:', target_max_length)
    # print("target_unique_text: ", target_unique_text)

    # ---------------Dictionary---------------#
    # Create token-index mapping
    target_token_index = dict([(char, i) for i, char in enumerate(target_unique_text)])
    print("target_token_index: ", target_token_index)
    # Reverse-lookup token index to decode sequences back to something readable.
    reverse_target_word_index = dict((i, word) for word, i in target_token_index.items())
    # print("reverse_target_char_index: ", reverse_target_word_index)

    """
    3. Manage Output Sentence Data //3.2. change the y words to index
    """

    with open(output_data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        index_sentence = 0

        target_texts = []
        y_train_len = []
        groundtruth_sentence = []

        for line in lines[: min(total_sentence_groundtruth, len(lines) - 1)]:  # get X rows of training instances
            print('line: ', line)

            groundtruth = line.split(' ')[1].split('（')[0]
            # print("groundtruth: ", groundtruth)

            target_text = line.split('（')[1][:-1]
            # print("target_text", target_text)                                       #for 1 sentence

            array_target_text = target_text.split(' ')  # change to array to find max length of words
            for j in range(len(array_target_text)):
                array_target_text[j] = target_token_index[array_target_text[j]]
            print("array_target_text: ", array_target_text)

            print("index_sentence list_num_samples_per_sentence", index_sentence, list_num_samples_per_sentence)
            for i in range(250):  # mapping with the # of VDO per sentence
                target_texts.append(array_target_text)
                #            print("array_target_text",array_target_text,len(array_target_text))
                y_train_len.append(len(array_target_text))
                groundtruth_sentence.append(groundtruth)
                # print("target",len(target_texts))
            # print("target_texts", target_texts)

            # print(target_texts)
            # print(y_train_len)
            # print(groundtruth_sentence)

            index_sentence += 1

    num_samples = len(target_texts)
    # print("target_texts: ", target_texts)
    # print("total target_texts samples: ", len(target_texts))
    y_train_pad = sequence.pad_sequences(target_texts, value=float(nb_labels), dtype='float16',
                                         padding="post")  # pad with the nb_labels
    y_test_pad = sequence.pad_sequences(target_texts, value=float(nb_labels), dtype='float16', padding="post")

    y_train_len = np.asarray(y_train_len)
    y_test_len = y_train_len

    print(y_train_pad)
    print(y_train_len)

    print(len(y_train_pad))
    print(len(y_train_len))

    return y_train_pad, y_train_len, y_test_pad, y_test_len,


max_frame = 210


def train(shuffle=False):
    y_train_pad, y_train_len, y_test_pad, y_test_len, = generate_y()
    x_train = generate_data_list()

    if shuffle:
        c = list(zip(y_train_pad, y_train_len, x_train))
        random.shuffle(c)
        y_train_pad, y_train_len, x_train = zip(*c)

    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(max_frame, 56, 56, 256))

    vgg = VGG(i_vgg)
    m_vgg = Model(inputs=[i_vgg], outputs=[vgg])

    # TCN
    o_tcn_full = TCN_layer(vgg, 5)
    tcn_model = Model(inputs=[i_vgg], outputs=[o_tcn_full])

    tcn_model.summary()
    exit(0)

    # # Full Frame Model
    # m_vgg.summary()

    ### The Custom Loop
    # The train_on_batch function
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam()

    # Compile
    m_vgg.compile(loss=loss, optimizer=optimizer)

    # def train_on_batch(X, y):
    #     with tf.GradientTape() as tape:
    #         ŷ = m_vgg(X, training=True)
    #         loss_value = loss(y, ŷ)
    #
    #     grads = tape.gradient(loss_value, m_vgg.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, m_vgg.trainable_weights))
    #
    # # The validate_on_batch function
    # def validate_on_batch(X, y):
    #     ŷ = m_vgg(X, training=False)
    #     loss_value = loss(y, ŷ)
    #     return loss_value

    # Putting it all together
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(0.001)
    batch_size = 1
    epochs = 1

    for epoch in range(0, epochs):
        for i in range(0, len(x_train) // batch_size):
            X = x_train[i * batch_size:min(len(x_train), (i + 1) * batch_size)]
            y = y_train_pad[i * batch_size:min(len(y_train_pad), (i + 1) * batch_size)]

            x_list = []
            y_list = []

            for i_data in range(0, len(X)):
                print(len(X))
                print(X[i_data])
                print(y[i_data])
                x_npz = np.load(X[i_data])
                print(x_npz['arr_0'].shape)

                x_list.append(np.asarray(x_npz['arr_0']))
                y_list.append(y[i_data])

            x_reshape = np.reshape(x_list, (1, 1, 210, 56, 56, 256))
            y_reshape = np.reshape(y_list, (1, 1, 6))

            print(np.asarray(x_list).shape)
            print(np.asarray(y_list).shape)

            m_vgg.train_on_batch(x_list[0], y_list[0])

        val_loss = []
        for i in range(0, len(x_train) // batch_size):
            X = x_train[i * batch_size:min(len(x_train), (i + 1) * batch_size)]
            y = y_train_pad[i * batch_size:min(len(y_train_pad), (i + 1) * batch_size)]
            val_loss.append(validate_on_batch(X, y))

        print('Validation Loss: ' + str(np.mean(val_loss)))

    # TCN Model
    o_tcn_full = TCN_layer(vgg, 5)

    # tcn_model = Model(inputs=[i_vgg], outputs=[o_tcn_full])
    # tcn_model = Dense(256, name='dense_o_tcn_full1')(tcn_model)
    #
    # o_tcn_intra_block1 = TCN_layer(tcn_model, 1)  # The TCN layers are here.
    #
    # tcn_model = Dense(256, name='dense_o_tcn_full1')(o_tcn_intra_block1)
    #
    # tcn_model.summary()

    # TCN
    #
    #     i_tcn1 = Input(name='inputtcn1', shape=(max_frame, 1536))
    #     # slice_full = tf.slice(i_tcn1,[0,0,0],[1,55,512])
    #     # slice_head = tf.slice(i_tcn1,[0,0,512],[1,55,256])
    #     # slice_hand = tf.slice(i_tcn1,[0,0,768],[1,55,512])
    #     # slice_full = Lambda(slice, arguments={'h': 0, 'w': 512})(i_tcn1)
    #     # slice_head = Lambda(slice, arguments={'h': 512, 'w': 768})(i_tcn1)
    #     # slice_hand = Lambda(slice, arguments={'h': 768, 'w': 1280})(i_tcn1)
    #     # if is_pose:
    #     #     slice_pose = Lambda(slice, arguments={'h': 1280, 'w': 1536})(i_tcn1)
    #
    #     # print(i_tcn1,slice_full,slice_head,slice_hand)
    #     # o_tcn_inter1 = TCN(return_sequences=True, activation='relu', dilations=[1, 2, 4, 8])(i_tcn1)
    #     o_tcn_inter1 = TCN_layer(i_tcn1, 5)
    #     # print("o_tcn_inter1",o_tcn_inter1)
    #     o_tcn_inter1 = Dense(768, name='dense_o_tcn_inter1')(o_tcn_inter1)
    #
    #     # o_tcn_full = TCN(return_sequences=True, activation='relu', dilations=[1, 2, 4, 8])(slice_full)
    #     o_tcn_full = TCN_layer(slice_full, 5)
    #     # print("o_tcn_full",o_tcn_full)
    #     o_tcn_full = Dense(256, name='dense_o_tcn_full1')(o_tcn_full)
    #
    #     # o_tcn_head = TCN(return_sequences=True, activation='relu', dilations=[1, 2, 4, 8])(slice_head)
    #     o_tcn_head = TCN_layer(slice_head, 5)
    #     # print("o_tcn_head",o_tcn_head)
    #     o_tcn_head = Dense(256, name='dense_o_tcn_head1')(o_tcn_head)
    #
    #     if is_pose:
    #         o_tcn_pose = TCN_layer(slice_pose, 5)
    #         # print("o_tcn_head",o_tcn_head)
    #         o_tcn_pose = Dense(256, name='dense_o_tcn_pose1')(o_tcn_pose)
    #
    #     # o_tcn_hand = TCN(return_sequences=True, activation='relu', dilations=[1, 2, 4, 8])(slice_hand)
    #     o_tcn_hand = TCN_layer(slice_hand, 5)
    #     # print("o_tcn_hand",o_tcn_hand)
    #     o_tcn_hand = Dense(256, name='dense_o_tcn_hand1')(o_tcn_hand)
    #
    #     # o_intra_block1 = tf.concat([o_tcn_head, o_tcn_hand, o_tcn_full],2)
    #     if is_pose:
    #
    #         o_intra_block1 = concatenate([o_tcn_head, o_tcn_hand, o_tcn_full, o_tcn_pose], axis=2)
    #     else:
    #         o_intra_block1 = concatenate([o_tcn_head, o_tcn_hand, o_tcn_full], axis=2)
    #     # o_intra_block1 = concatenate([o_tcn_head, o_tcn_ha1nd],axis=2)
    #
    #     o_tcn_intra_block1 = TCN_layer(o_intra_block1, 1)  # The TCN layers are here.
    #     # print("o_tcn_intra_block1",o_tcn_intra_block1)
    #     if is_pose:
    #         o_tcn_intra_block1 = Dense(1024, name='dense_o_tcn_intra_block1')(o_tcn_intra_block1)
    #     else:
    #         o_tcn_intra_block1 = Dense(768, name='dense_o_tcn_intra_block1')(o_tcn_intra_block1)
    #
    #     o_inter_block1 = concatenate([o_tcn_inter1, o_tcn_intra_block1], axis=2, name='o_tcn_inter_block1')
    #     # o_inter_block1 = tf.concat([o_tcn_inter1, o_tcn_intra_block1],2)
    #     if is_pose:
    #         o_inter_block1 = Dense(1024)(o_inter_block1)
    #     else:
    #         o_inter_block1 = Dense(768)(o_inter_block1)
    #
    #     # print(o_inter_block1, o_intra_block1)
    #     inter_block1 = MaxPooling1D(pool_size=tcn_kernel, strides=tcn_stride)(
    #         o_inter_block1)  # divided by 2
    #     intra_block1 = MaxPooling1D(pool_size=tcn_kernel, strides=tcn_stride)(
    #         o_intra_block1)  # divided by 2
    #
    #     # o_block1 = Concatenate([inter_block1, o_tcn_intra_block1],2)
    #
    #     m_tcn1 = Model(inputs=[i_tcn1], outputs=[inter_block1, intra_block1])

    print('train')


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


def VGG(i_vgg):
    #    input_data = Input(name='input', shape=(None,224, 224, 3), dtype = "float16")
    # Izda.add(TimeDistributed(
    #    Convolution2D(40,3,3,border_mode='same'), input_shape=(sequence_lengths, 1,8,10)))
    #    model = Sequential()
    # Izda.add(TimeDistributed(
    #    Convolution2D(40,3,3,border_mode='same'), input_shape=(sequence_lengths, 1,8,10)))

    # #    i_vgg = tf.keras.layers.Input(batch_shape=(None,55,224,224,3))
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
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", name='block3_conv1'))(model)
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


if __name__ == '__main__':
    # generate_data_list()
    train()
    #
    # generate_y()
