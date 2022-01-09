import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import cv2
import gc
import os
from numpy import savez_compressed
import tensorflow as tf
from tensorflow.python.keras.models import Model

from extract_layer4 import get_output_layer

from sklearn.preprocessing import LabelEncoder
import functools

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as ff
# from pose_hrnet import get_pose_net
from pose_higher_hrnet import get_pose_net
# from hrnet_dekr import get_pose_net
# import coremltools as ct
from collections import OrderedDict
# from config import cfg
from config_higher import cfg
from config import update_config

from jiwer import wer

from utils import pose_process, plot_pose

from tensorflow.python.keras.layers import Conv1D, Add, Activation, Dense, TimeDistributed, Conv2D, \
    MaxPooling2D, GlobalAveragePooling2D, LSTM, MaxPooling1D, Bidirectional
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.backend import concatenate
from keras_ctcmodel.CTCModel import CTCModel as CTCModel

from tensorflow.keras.utils import Progbar

from os import listdir
from os.path import isfile, join

# from natsort import natsorted

multi_scales = [512, 640]

print(torch.__version__)
print(torch.cuda.is_available())

CSV_PATH = r"D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\manual"
TRAIN_CSV = r'\train.corpus.csv'
DEV_CSV = r'\dev.corpus.csv'
TEST_CSV = r'\test.corpus.csv'

TRAIN_PATH = r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\train\\'
DEV_PATH = r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\dev\\'
TEST_PATH = r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\test\\'

# TRAIN_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\train\\'
# DEV_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\dev\\'
# TEST_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\test\\'

TRAIN_KEYPOINT_PATH = r'D:\Dataset\Sign Language\Phoenix\Keypoint\train\\'
DEV_KEYPOINT_PATH = r'D:\Dataset\Sign Language\Phoenix\Keypoint\dev\\'
TEST_KEYPOINT_PATH = r'D:\Dataset\Sign Language\Phoenix\Keypoint\test\\'

MODEL_SAVE_PATH = r"D:\Dataset\Sign Language\Phoenix\Saved-Model"
MODEL_LOAD_PATH = r"D:\Dataset\Sign Language\Phoenix\Load"

TRAIN_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-default\train\\'
DEV_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-default\dev\\'
TEST_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-default\test\\'

DATA_PATH = r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\\'

IS_ENDTOEND = True

PREVIEW = False
DEBUG = False
TESTING = False

LOAD_WEIGHT = False

CONFIG = 'dev'

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

index_mirror = np.concatenate([
    [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
    [21, 22, 23, 18, 19, 20],
    np.arange(40, 23, -1), np.arange(50, 40, -1),
    np.arange(51, 55), np.arange(59, 54, -1),
    [69, 68, 67, 66, 71, 70], [63, 62, 61, 60, 65, 64],
    np.arange(78, 71, -1), np.arange(83, 78, -1),
    [88, 87, 86, 85, 84, 91, 90, 89],
    np.arange(113, 134), np.arange(92, 113)
]) - 1
assert (index_mirror.shape[0] == 133)

selected = np.concatenate(([0, 5, 6, 7, 8, 9, 10],
                          [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                          [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),  # 27


def norm_numpy_totensor(img):
    img = img.astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]
    return torch.from_numpy(img).permute(0, 3, 1, 2)


def stack_flip(img):
    img_flip = cv2.flip(img, 1)
    return np.stack([img, img_flip], axis=0)


def merge_hm(hms_list):
    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1, :, :, :] = torch.flip(hms[1, index_mirror, :, :], [2])

    hm = torch.cat(hms_list, dim=0)
    # print(hm.size(0))
    hm = torch.mean(hms, dim=0)
    return hm


def load_img(path):
    allframes = [path + r'\\' + f for f in listdir(path) if isfile(join(path, f))]

    if DEBUG:
        print(allframes)

    video_array = []
    for img in allframes:
        frame = Image.open(img)
        resized_image = frame.resize((224, 224))
        video_array.append(np.array(resized_image))

    return np.array(video_array)


def extract_data(config):
    csv_source = ""
    extract_path = ""
    source_path = ""

    if config == 'dev':
        source_path = DEV_PATH
        csv_source = DEV_CSV
        extract_path = DEV_EXTRACT_PATH
    elif config == 'train':
        source_path = TRAIN_PATH
        csv_source = TRAIN_CSV
        extract_path = TRAIN_EXTRACT_PATH
    elif config == 'test':
        source_path = TEST_PATH
        csv_source = TEST_CSV
        extract_path = TEST_EXTRACT_PATH

    phoenix_csv = pd.read_csv(CSV_PATH + csv_source, sep='|', header=None, skiprows=1)

    # Creating a Dataframe
    df = pd.DataFrame(phoenix_csv)

    # show the dataframe
    if DEBUG:
        print(df.head())
        print(df[3].head())

    all_sentences = df[3].unique().tolist()

    sentences = []
    count = 0
    for col in all_sentences:
        words = str(col).split()
        sentences.append(words)
        if DEBUG:
            print(words)
            print(count)

        count += 1


    target_tokens = list(
        functools.reduce(lambda a, b: a.union(b), list(map(lambda x: set(x), sentences))))
    # target_tokens += ['#START', '#END']

    sentences_array = list(map(lambda x: set(x), sentences))
    # max_sentence = len(max(sentences_array, key=lambda x: len(x))) + 2
    max_sentence = len(max(sentences_array, key=lambda x: len(x)))
    num_classes = len(target_tokens)

    le = LabelEncoder()
    le.fit(target_tokens)

    print(max_sentence)
    print(num_classes)
    print(target_tokens)

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    x_data = df[0].tolist()
    x_data_path = df[1].tolist()

    print(x_data)
    print(x_data_path)

    PREFIX_FOLDER = r'\\1\\'

    for idx, val in enumerate(x_data_path):
        data_path = source_path + x_data[idx] + PREFIX_FOLDER
        if DEBUG:
            print(data_path)

        save_file = f'{extract_path}\{x_data[idx]}.npz'

        if os.path.isfile(save_file):
            print('file exist')
            continue

        allframes = [data_path + f for f in listdir(data_path) if isfile(join(data_path, f))]

        if DEBUG:
            print(allframes)

        video_array = []
        for img in allframes:
            frame = Image.open(img)
            resized_image = frame.resize((224, 224))
            video_array.append(np.asarray(resized_image))

        output = get_output_layer(src=np.asarray(video_array))
        tf.keras.backend.clear_session()
        gc.collect()

        if DEBUG:
            print(output.shape)

        # save_dir = f'{extract_path}\{x_data[idx]}'

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        savez_compressed(save_file, output)

        # print(video_array)
        # exit()

    # print(le_name_mapping)
    # print(le.transform(sentences[0]))
    # print(le.inverse_transform(le.transform(sentences[0])))

    # # iterating the columns
    # for col in gsl_all.row:
    #     print(col)


def extract_keypoint(config):
    csv_source = ""
    extract_path = ""
    source_path = ""

    if config == 'dev':
        source_path = DEV_PATH
        csv_source = DEV_CSV
        extract_path = DEV_KEYPOINT_PATH
    elif config == 'train':
        source_path = TRAIN_PATH
        csv_source = TRAIN_CSV
        extract_path = TRAIN_KEYPOINT_PATH
    elif config == 'test':
        source_path = TEST_PATH
        csv_source = TEST_CSV
        extract_path = TEST_KEYPOINT_PATH

    phoenix_csv = pd.read_csv(CSV_PATH + csv_source, sep='|', header=None, skiprows=1)

    # Creating a Dataframe
    df = pd.DataFrame(phoenix_csv)

    # show the dataframe
    if DEBUG:
        print(df.head())
        print(df[3].head())

    x_data = df[0].tolist()
    x_data_path = df[1].tolist()

    print(x_data)
    print(x_data_path)

    from os import listdir
    from os.path import isfile, join

    PREFIX_FOLDER = r'\\1\\'

    with torch.no_grad():
        # config = 'wholebody_w48_384x288.yaml'
        config = 'w48_640_adam_lr1e-3.yaml'
        cfg.merge_from_file(config)

        newmodel = get_pose_net(cfg, is_train=False)
        # print(newmodel)

        checkpoint = torch.load(
            # r'D:\Dataset\Sign Language\HRnet\hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth')
            # r'D:\Dataset\Sign Language\HRnet\pose_hrnet_w48_384x288.pth')
            # r'D:\Dataset\Sign Language\HRnet\pose_dekr_hrnetw48_coco.pth')
            # 'D:\Dataset\Sign Language\HRnet\pose_higher_hrnet_w32_512.pth')
            'D:\Dataset\Sign Language\HRnet\higher_hrnet48_coco_wholebody_512x512_plus-934f08aa_20210517.pth')

        for key in checkpoint['state_dict']:
            print(key)

        # print(checkpoint['state_dict'])
        # # print(checkpoint)
        # exit()

        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            if 'backbone.' in k:
                name = k[9:]  # remove module.
            if 'keypoint_head.' in k:
                name = k[14:]  # remove module.
            # if 'final_layers.' in k:
            #     name = k[13:]  # remove module.
            #     state_dict.items()
            # if 'deconv_layers.' in k:
            #     name = k[14:]  # remove module.

            # print(name)
            # exit()
            new_state_dict[name] = v
        newmodel.load_state_dict(new_state_dict)

        newmodel.cuda().eval()

        for idx, val in enumerate(x_data_path):
            data_path = source_path + x_data[idx] + PREFIX_FOLDER
            if DEBUG:
                print(data_path)

            save_file = f'{extract_path}\{x_data[idx]}.npy'

            # if os.path.isfile(save_file):
            #     print('file exist')
            #     continue

            allframes = [data_path + f for f in listdir(data_path) if isfile(join(data_path, f))]

            if DEBUG:
                print(allframes)

            output_list = []
            for i, img in enumerate(allframes):
                frame = Image.open(img)

                img = ImageOps.mirror(frame)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                frame_height, frame_width = img.shape[:2]
                img = cv2.flip(img, flipCode=1)
                out = []
                for scale in multi_scales:
                    if scale != 512:
                        img_temp = cv2.resize(img, (scale, scale))
                    else:
                        img_temp = img
                    img_temp = stack_flip(img_temp)
                    img_temp = norm_numpy_totensor(img_temp).cuda()
                    hms = newmodel(img_temp)
                    if scale != 512:
                        out.append(ff.interpolate(hms, (frame_width // 4, frame_height // 4), mode='bilinear'))
                    else:
                        out.append(hms)

                out = merge_hm(out)
                result = out.reshape((133, -1))
                result = torch.argmax(result, dim=1)
                result = result.cpu().numpy().squeeze()

                y = result // (frame_width // 4)
                x = result % (frame_width // 4)
                pred = np.zeros((133, 3), dtype=np.float32)
                pred[:, 0] = x
                pred[:, 1] = y

                hm = out.cpu().numpy().reshape((133, frame_height // 4, frame_height // 4))

                pred = pose_process(pred, hm)
                pred[:, :2] *= 4.0
                assert pred.shape == (133, 3)

                output_list.append(pred)

                img = np.asarray(img)
                for j in range(133):
                    img = cv2.circle(img, (int(x[j]), int(y[j])), radius=2, color=(255,0,0), thickness=-1)
                img = plot_pose(img, pred)
                cv2.imwrite(r'D:\Dataset\Sign Language\Phoenix\Output\{}.png'.format(x_data[i]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                # writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            output_list = np.array(output_list)

            # np.save(save_file, output_list)

            tf.keras.backend.clear_session()
            gc.collect()


def get_frame_length(config):
    csv_source = ""
    extract_path = ""
    source_path = ""

    if config == 'dev':
        source_path = DEV_PATH
        csv_source = DEV_CSV
        extract_path = DEV_KEYPOINT_PATH
    elif config == 'train':
        source_path = TRAIN_PATH
        csv_source = TRAIN_CSV
        extract_path = TRAIN_KEYPOINT_PATH
    elif config == 'test':
        source_path = TEST_PATH
        csv_source = TEST_CSV
        extract_path = TEST_KEYPOINT_PATH

    phoenix_csv = pd.read_csv(CSV_PATH + csv_source, sep='|', header=None, skiprows=1)

    # Creating a Dataframe
    df = pd.DataFrame(phoenix_csv)

    # show the dataframe
    if DEBUG:
        print(df.head())
        print(df[3].head())

    x_data = df[0].tolist()
    x_data_path = df[1].tolist()

    from os import listdir
    from os.path import isfile, join

    PREFIX_FOLDER = r'\\1\\'

    all_length = []

    for idx, val in enumerate(x_data_path):
        data_path = source_path + x_data[idx] + PREFIX_FOLDER

        if DEBUG:
            print(data_path)

        allframes = [data_path + f for f in listdir(data_path) if isfile(join(data_path, f))]

        if DEBUG:
            print(allframes)

        all_length.append(len(allframes))

    return all_length


def get_all_length():
    FRAME_LENGTH_PATH = r'D:\Dataset\Sign Language\Phoenix\Frame-length\\'

    # np.save(FRAME_LENGTH_PATH + 'dev.npy', np.array(get_frame_length('dev')))
    # np.save(FRAME_LENGTH_PATH + 'test.npy', np.array(get_frame_length('test')))
    # np.save(FRAME_LENGTH_PATH + 'train.npy', np.array(get_frame_length('train')))

    dev = np.load(FRAME_LENGTH_PATH + 'dev.npy')
    test = np.load(FRAME_LENGTH_PATH + 'test.npy')
    train = np.load(FRAME_LENGTH_PATH + 'train.npy')

    temp = np.concatenate((dev, test, train), axis=0)

    padding_x = np.max(temp)

    return dev, test, train, padding_x


def get_label_data(config='dev'):
    csv_source = ""
    extract_path = ""
    keypoint_path = ""

    if config == 'dev':
        csv_source = DEV_CSV
        extract_path = DEV_EXTRACT_PATH
        keypoint_path = DEV_KEYPOINT_PATH
    elif config == 'train':
        csv_source = TRAIN_CSV
        extract_path = TRAIN_EXTRACT_PATH
        keypoint_path = TRAIN_KEYPOINT_PATH
    elif config == 'test':
        csv_source = TEST_CSV
        extract_path = TEST_EXTRACT_PATH
        keypoint_path = TEST_KEYPOINT_PATH

    phoenix_csv = pd.read_csv(CSV_PATH + csv_source, sep='|', header=None, skiprows=1)

    # Creating a Dataframe
    df = pd.DataFrame(phoenix_csv)

    # show the dataframe
    if DEBUG:
        print(df.head())
        print(df[3].head())

    x_data = df[0].tolist()
    x_data_path = df[1].tolist()

    x_list = []
    x_key_list = []
    signer_list = df[2].tolist()
    sentence_list = df[3].tolist()

    for idx, val in enumerate(x_data_path):
        x_key_path = f'{keypoint_path}\{x_data[idx]}.npy'

        if IS_ENDTOEND:
            x_path = f'{DATA_PATH}\{config}\{x_data[idx]}\{"1"}'
        else:
            x_path = f'{extract_path}\{x_data[idx]}.npz'


        x_list.append(x_path)
        x_key_list.append(x_key_path)

    return x_list, x_key_list, signer_list, sentence_list


def get_sentence_token():
    a, b, c, sentence_dev = get_label_data('dev')
    a, b, c, sentence_test = get_label_data('test')
    a, b, c, sentence_train = get_label_data('train')

    all_sentences = np.concatenate((sentence_dev, sentence_test, sentence_train), axis=0)

    sentences = []
    count = 0
    for col in all_sentences:
        words = str(col).split()
        sentences.append(words)
        if DEBUG:
            print(words)
            print(count)

        count += 1

    target_tokens = list(
        functools.reduce(lambda a, b: a.union(b), list(map(lambda x: set(x), sentences))))
    # target_tokens += ['#START', '#END']

    sentences_array = list(map(lambda x: set(x), sentences))
    # max_sentence = len(max(sentences_array, key=lambda x: len(x))) + 2
    max_sentence = len(max(sentences_array, key=lambda x: len(x)))
    num_classes = len(target_tokens)

    le = LabelEncoder()
    le.fit(target_tokens)

    print(max_sentence)
    print(num_classes)
    print(target_tokens)

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    print(le_name_mapping)

    sentence_dev = list(map(lambda x: str(x).split(), sentence_dev))
    sentence_test = list(map(lambda x: str(x).split(), sentence_test))
    sentence_train = list(map(lambda x: str(x).split(), sentence_train))

    transformed_dev = list(map(lambda x: le.transform(x), sentence_dev))
    transformed_test = list(map(lambda x: le.transform(x), sentence_test))
    transformed_train = list(map(lambda x: le.transform(x), sentence_train))

    label_len_dev = list(map(lambda x: len(str(x).split()), sentence_dev))
    label_len_test = list(map(lambda x: len(le.transform(x)), sentence_test))
    label_len_train = list(map(lambda x: len(le.transform(x)), sentence_train))

    if DEBUG:
        print(transformed_dev)
        print(transformed_test)
        print(transformed_train)

        print(label_len_dev)
        print(label_len_test)
        print(label_len_train)


    # print(le.transform(sentences))
    # print(le.inverse_transform(le.transform(sentence_dev)))

    transformed_dev = tf.keras.preprocessing.sequence.pad_sequences(
        transformed_dev, padding="post"
    )

    transformed_test = tf.keras.preprocessing.sequence.pad_sequences(
        transformed_test, padding="post"
    )

    transformed_train = tf.keras.preprocessing.sequence.pad_sequences(
        transformed_train, padding="post"
    )

    return transformed_dev, transformed_test, transformed_train, label_len_dev, label_len_test, label_len_train


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

def VGG_2(i_vgg):
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

def SpatialBlock(input):

    resnet = tf.keras.applications.ResNet50V2(weights=r'D:\Dataset\Sign Language\ResNet\resnet50v2_weights_tf_dim_ordering_tf_kernels.h5')

    intermediate_layer_model = Model(inputs=resnet.input, outputs=resnet.get_layer('conv5_block3_1_conv').output, name='intermediate_resnet')

    intermediate_layer_model.summary()

    model1 = TimeDistributed(intermediate_layer_model, name='time_distributed_resnet')(input)

    model1 = TimeDistributed(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", name='block3_conv1'))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", name='block3_conv2'))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", name='block3_conv3'))(model1)
    model1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name='block4_conv1'))(model1)
    model1 = TimeDistributed(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", name='block4_conv2'))(model1)

    model = TimeDistributed(GlobalAveragePooling2D(name="global_max_full"))(model1)

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


def train_ctc():
    dev_framelen, test_framelen, train_framelen, padding_x = get_all_length()

    x_list, x_key_list, signer_list, sentence_list = get_label_data('train')

    x_list_test, x_key_list_test, signer_list_test, sentence_list_test = get_label_data('test')

    transformed_dev, transformed_test, transformed_train, label_len_dev, label_len_test, label_len_train = get_sentence_token()


    x_data = x_list
    x_data_keypoint = x_key_list
    y_data = transformed_train
    x_len = train_framelen
    y_len = label_len_train

    x_data_val = x_list_test
    x_data_keypoint_validate = x_key_list_test
    y_data_val = transformed_test
    x_len_val = test_framelen
    y_len_val = label_len_test

    max_len = padding_x
    num_classes = 1295
    # skipped_max_len

    # if shuffle:
    #     c = list(zip(x_data, y_data))
    #     random.shuffle(c)
    #     x_data, y_data = zip(*c)

    # Input from intermediate layer
    i_vgg = tf.keras.Input(name="input_1", shape=(max_len, 224, 224, 3))

    spatialBlock = SpatialBlock(i_vgg)

    attn_spatial = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=4, name="spatial_attn")(spatialBlock, spatialBlock)

    # Input Keypoint
    i_keypoint = tf.keras.Input(name="input_1_keypoint", shape=(max_len, 1, 27, 3))
    output_keypoint = TimeDistributed(GlobalAveragePooling2D(name="global_max_full"))(i_keypoint)
    dense_input_keypoint = Dense(256, activation='relu', name='dense_keypoint')(output_keypoint)

    attn_keypoint = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=4, name="keypoint_attn")(dense_input_keypoint, dense_input_keypoint)

    '''
    TCN -> Dense
    '''
    o_tcn_full = TCN_layer(attn_spatial, 5)
    o_tcn_keypoint = TCN_layer(attn_keypoint, 5)

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
        network.load_model(path_dir=f'{MODEL_LOAD_PATH}', file_weights='/model_weights.hdf5',
                           optimizer=Adam(0.00001), init_last_layer=False, init_archi=False)
        print('Weight Loaded from previous train')

    network.compile(optimizer=Adam(lr=0.00001))

    network.summary()

    metrics_names = ['val']

    batch_size = 1
    epochs = 30

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
            length = 72

            # print(f'\nLoad data {epoch} / batch {i}')
            for i_data in range(0, len(X)):
                # x_seq, y_seq = sentence_sequence_generator_npz(X[i_data], y[i_data])

                # load_npz = np.load(X[i_data])

                load_npz = load_img(X[i_data])

                if DEBUG:
                    print(X[i_data])
                    print(X_keypoint[i_data])
                    print(X_len[i_data])

                load_npz = np.pad(load_npz, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]

                x_npy = x_npy.reshape((x_npy.shape[0], 1, 27, 3))
                x_npy = np.pad(x_npy, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))
                x_key_list.append(np.asarray(x_npy))

                x_list.append(load_npz)

                label = y[i_data]

                y_list.append(label)

                # Hardcoded ???
                x_len_list.append(length)
                y_len_list.append(np.asarray(Y_len[i_data]))

                Y_zeros_list.append(Y_zeros[i_data])

                if DEBUG:
                    print(x_npy.shape)
                    print(load_npz.shape)

            input = [np.array(x_list), np.array(x_key_list), np.array(y_list), np.array(x_len_list),
                     np.array(y_len_list)]

            history = network.train_on_batch(
                x=input,
                y=np.array(Y_zeros_list))

            values = [('val', history)]

            pb_i.add(batch_size, values=values)
            loss.append(history)

        loss_avg = np.average(loss)
        print('#########')
        print(f'Loss AVG / EPOCH {epoch + 1} : {loss_avg}')
        print('#########')

        if loss_avg < loss_:
            loss_ = loss_avg
            network.save_model(f'{MODEL_SAVE_PATH}/')


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

                # load_npz = np.load(X[i_data])

                load_npz = load_img(X[i_data])

                load_npz = np.pad(load_npz, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

                x_npy = np.load(X_keypoint[i_data])
                x_npy = x_npy[:, selected, :]

                x_npy = x_npy.reshape((x_npy.shape[0], 1, 27, 3))
                x_npy = np.pad(x_npy, ((0, max_len - X_len[i_data]), (0, 0), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))
                x_key_list.append(np.asarray(x_npy))

                x_list.append(load_npz)
                # y_list.append(np.asarray(y[i_data]))

                # x_len_list.append(np.asarray(X_len[i_data]))

                label = y[i_data]

                y_list.append(label)

                # Hardcoded ???
                x_len_list.append(length)
                y_len_list.append(np.asarray(Y_len[i_data]))

                Y_zeros_list.append(Y_zeros[i_data])

            predict = network.predict_on_batch(
                x=[np.concatenate((np.array(x_list), np.array(x_list)), axis=0), np.concatenate((np.array(x_key_list), np.array(x_key_list)), axis=0),
                   np.concatenate((np.array(x_len_list), np.array(x_len_list)), axis=0)])
            y_new = np.concatenate((np.array(y_list), np.array(y_list)), axis=0)

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

    print(acc_model)
    print(loss_model)

    print('#########')
    # print(f'model WER : {error_avg}')
    print('######### xoxo #########')
    print('train done')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # extract_data('test')
    # extract_keypoint('train')
    # dev, test, train, padding_x = get_all_length()
    #
    # print(dev)
    # print(test)
    # print(train)
    # print(padding_x)

    # x_list, x_key_list, signer_list, sentence_list = get_label_data('dev')
    #
    # print(x_list)
    # print(x_key_list)
    # print(signer_list)
    # print(sentence_list)

    # transformed_dev, transformed_test, transformed_train, label_len_dev, label_len_test, label_len_train = get_sentence_token()

    train_ctc()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
