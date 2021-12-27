import pandas as pd
import numpy as np
from PIL import Image
import cv2
import gc
import os
from numpy import savez_compressed
import tensorflow as tf

from extract_layer4 import get_output_layer

from sklearn.preprocessing import LabelEncoder
import functools

CSV_PATH = r"F:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\manual"
TRAIN_CSV = r'\train.corpus.csv'
DEV_CSV = r'\dev.corpus.csv'
TEST_CSV = r'\test.corpus.csv'

TRAIN_PATH = r'F:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\train\\'
DEV_PATH = r'F:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\dev\\'
TEST_PATH = r'F:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\test\\'

TRAIN_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\train\\'
DEV_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\dev\\'
TEST_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\test\\'

PREVIEW = False
DEBUG = False
TESTING = False

CONFIG = 'dev'


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

    # print(list(sentences).item)

    # exit()

    # mapped = map(lambda x: set(x), sentences)
    # print(list(mapped))
    # exit()

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

    from os import listdir
    from os.path import isfile, join

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    extract_data('train')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
