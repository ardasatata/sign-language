import math
import os
import json
import random
import zipfile

import cv2
import numpy as np
from keras.optimizer_v2.adam import Adam
from matplotlib.backends.backend_template import FigureCanvas
from numpy import savez_compressed, savez
import tensorflow as tf
# import tensorflow.keras as keras
import progressbar
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

from mpl_toolkits import mplot3d

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    # data = np.load(r'F:\Dataset\Sign Language\AUTSL\data\out\train\npy3\signer12_sample1056_color.mp4.npy')
    data = np.load(r'F:\Dataset\Sign Language\WLASL-Alter\Key\07075.mp4.npy')

    selected = selected_joints['27']

    skel = data[:, selected, :]

    # print(len(data))
    # print(data[0])

    print(skel.shape)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for frame in skel:
        xdata = []
        ydata = []
        zdata = []
        for key in frame:
            # print(key[0])

            xdata.append(key[0])
            ydata.append(key[1])
            zdata.append(key[2])

        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='jet');
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('prev', img)
        cv2.waitKey(100)

    # print(skel.shape[0])
