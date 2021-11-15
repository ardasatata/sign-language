import glob
import os
from os import listdir
from os.path import isfile, join

import gc

import tensorflow as tf

from tensorflow.keras.models import Model
from keras_preprocessing.image import load_img, img_to_array

gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)

from PIL import Image
from numpy import savez_compressed

import numpy as np

import cv2

MODEL_NAME = r"C:\Users\minelab\dev\TSL\model\model_3_point_full_data.h5"
# MODEL_NAME = r"D:\WLASL\model\TMC_VIDEO_WLASL_2000_10.830657319324773.h5"
# MODEL_NAME = r"D:\WLASL\model\TMC_VIDEO_WLASL_300.h5"

CLASS_NAME = r'\000009'

TARGET_DIR = r"G:\TSL\Processed-Dataset\Color-Cropped" + CLASS_NAME
DEST_DIR = r"D:\TSL\OUT_Layer_4" + CLASS_NAME + r"\\"

LOAD_IMG = 'test4.jpg'


def get_value(DIR):
    folders = [f.path for f in os.scandir(DIR) if f.is_dir()]

    onlyfiles = [f for f in listdir(DIR) if isfile(join(DIR, f))]

    print(onlyfiles)

    filePaths = glob.glob(DIR + "/*")

    for file in filePaths:
        # Save NPZ disini
        try:
            cap = cv2.VideoCapture(file)

            video = []
            # print(file[46:-4])
            # exit(0)

            while cap.isOpened():
                ret, frame = cap.read()
                # (height, width) = frame.shape[:2]
                if ret:
                    # cv2.imshow('vid', frame)

                    resized_image = cv2.resize(frame, (224, 224))

                    # APPEND Output layer 4 disini
                    video.append(np.asarray(resized_image))

                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else:
                    break

            # cv2.waitKey(125)

            print(np.asarray(video).shape)

            output = get_output_layer(src=np.asarray(video))
            tf.keras.backend.clear_session()
            gc.collect()
            print(output.shape)
            print(file[46:-4])

            savez_compressed(DEST_DIR + file[46:-4] + '.npz', output)

            exit(0)

            cap.release()

        except cv2.error as e:
            print(e)
            False

        tf.keras.backend.clear_session()

        print('save npz')

# block3_conv1
# conv2_block3_preact_relu

def get_output_layer(src=LOAD_IMG, layer_name='block3_conv1'):
    # model = tf.keras.models.load_model(MODEL_NAME)

    # model = tf.keras.applications.ResNet50V2(weights='imagenet')
    model = tf.keras.applications.VGG16(weights="imagenet")

    model.load_weights(filepath=MODEL_NAME, by_name=True)

    # model.summary()
    #
    # exit()

    # img = load_img(img_path, target_size=(224, 224))

    # img = Image.fromarray(src)
    # img = img.resize(size=(224, 224))
    #
    # img_arr = np.asarray(img)
    #
    # arr = [img_arr]
    #
    # imgarr = np.array(arr)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(src)
    # print(intermediate_output[0].shape)
    # (56, 56, 256)

    return intermediate_output


def test_model():
    model = tf.keras.models.load_model(MODEL_NAME)

    model.summary()

    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.get_layer[0].output)

    print(model.get_layer(index=7).output)
    exit(0)


if __name__ == '__main__':
    # test_model()
    get_value(TARGET_DIR)
