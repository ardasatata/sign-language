import tensorflow as tf

import keras

# print(tf.version.VERSION)
# print(keras.__version__)

import numpy as np

x_npz = np.load(r"F:\Dataset\Sign Language\Demo CSL\Temp\11_05_2021_10_48_14.npz", mmap_mode='r')

print(x_npz['arr_0'])
print(np.asarray(x_npz['arr_0']).shape)

import cv2

# cap = cv2.VideoCapture(r'F:\Dataset\Sign Language\Demo CSL\Temp\11_08_2021_15_21_09.mp4')
# cap.set(cv2.CAP_PROP_FPS, int(30))
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps    = cap.get(cv2.CAP_PROP_FPS)
#
# print(length)
# print(width)
# print(height)
# print(fps)

