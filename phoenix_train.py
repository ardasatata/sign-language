import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import cv2
import gc
import os
from numpy import savez_compressed
import tensorflow as tf

from extract_layer4 import get_output_layer

from sklearn.preprocessing import LabelEncoder
import functools

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as ff
from pose_hrnet import get_pose_net
# import coremltools as ct
from collections import OrderedDict
from config import cfg
from config import update_config

from utils import pose_process, plot_pose
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

TRAIN_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\train\\'
DEV_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\dev\\'
TEST_EXTRACT_PATH = r'D:\Dataset\Sign Language\Phoenix\Extract_ResNet-conv5_block3_1_conv\test\\'

TRAIN_KEYPOINT_PATH = r'D:\Dataset\Sign Language\Phoenix\Keypoint\train\\'
DEV_KEYPOINT_PATH = r'D:\Dataset\Sign Language\Phoenix\Keypoint\dev\\'
TEST_KEYPOINT_PATH = r'D:\Dataset\Sign Language\Phoenix\Keypoint\test\\'

PREVIEW = False
DEBUG = False
TESTING = False

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
        config = 'wholebody_w48_384x288.yaml'
        cfg.merge_from_file(config)

        newmodel = get_pose_net(cfg, is_train=False)
        # print(newmodel)

        checkpoint = torch.load(
            r'D:\Dataset\Sign Language\HRnet\hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth')

        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'backbone.' in k:
                name = k[9:]  # remove module.
            if 'keypoint_head.' in k:
                name = k[14:]  # remove module.
            new_state_dict[name] = v
        newmodel.load_state_dict(new_state_dict)

        newmodel.cuda().eval()

        for idx, val in enumerate(x_data_path):
            data_path = source_path + x_data[idx] + PREFIX_FOLDER
            if DEBUG:
                print(data_path)

            save_file = f'{extract_path}\{x_data[idx]}.npy'

            if os.path.isfile(save_file):
                print('file exist')
                continue

            allframes = [data_path + f for f in listdir(data_path) if isfile(join(data_path, f))]

            if DEBUG:
                print(allframes)

            output_list = []
            for img in allframes:
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

            output_list = np.array(output_list)

            np.save(save_file, output_list)

            tf.keras.backend.clear_session()
            gc.collect()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # extract_data('train')
    extract_keypoint('train')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
