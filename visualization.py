import sys

from keras_preprocessing.image import load_img, img_to_array
from numpy import expand_dims
from tensorflow import keras
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.models import Model

import tensorflow as tf

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

import matplotlib.pyplot as plt

import numpy as np

import cv2

# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

import generate_pixel_map

np.set_printoptions(threshold=sys.maxsize)

MODEL_NAME = r"D:\TSL\model-pixel.h5"
# MODEL_NAME = r"C:\Users\minelab\dev\TSL\model\LSA64\model_pixel_20Class0.05101403589244.h5"
# MODEL_NAME = r"C:\Users\minelab\dev\TSL\model\model_3_point_full_data.h5"
# MODEL_NAME = r"D:\WLASL\model\TMC_VIDEO_WLASL_100.h5"
MODEL_NAME = r"D:\WLASL\model\TMC_VIDEO_WLASL_300.h5"


def visualize():
    # summarize feature map size for each conv layer
    from keras.applications.vgg16 import VGG16
    from matplotlib import pyplot
    # load the model
    model = keras.models.load_model(MODEL_NAME)
    # summarize feature map shapes
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)

    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[1].output)

    # load the image with the required shape
    img = load_img('origin.jpg', target_size=(112, 112))

    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)

    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)

    # get feature map for first hidden layer
    feature_maps = model.predict(img)

    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()


def visualize_all():
    # load the model
    model = tf.keras.applications.ResNet50V2(weights='imagenet')

    # model.load_weights(MODEL_NAME, by_name=True)

    model.summary()
    # exit(0)

    # redefine model to output right after the first hidden layer
    ixs = [6]
    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)
    # load the image with the required shape
    img = load_img(IMG_NAME, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot the output from each block
    square = 8
    for fmap in feature_maps:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[:, :, ix - 1], cmap='jet')
                ix += 1
        # show the figure
        pyplot.show()

    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)

    # Image titles
    image_titles = ['1']

    # Load images
    img1 = load_img(IMG_NAME, target_size=(224, 224))
    # img2 = load_img('images/bear.jpg', target_size=(224, 224))
    # img3 = load_img('images/soldiers.jpg', target_size=(224, 224))
    images = np.asarray([np.array(img1)])

    # Preparing input data
    X = preprocess_input(images)

    # Rendering
    subplot_args = {'nrows': 1, 'ncols': 2, 'figsize': (9, 3),
                    'subplot_kw': {'xticks': [], 'yticks': []}}
    # f, ax = plt.subplots(**subplot_args)
    # for i, title in enumerate(image_titles):
    #     ax[i].set_title(title, fontsize=14)
    #     ax[i].imshow(images[i])
    # plt.tight_layout()
    # plt.show()

    # Generate saliency map
    saliency_map = saliency(loss, X)
    saliency_map = normalize(saliency_map)

    # Render
    f, ax = plt.subplots(**subplot_args)
    ax[0].set_title('map', fontsize=14)
    ax[0].imshow(saliency_map[0], cmap='jet')
    ax[1].set_title('img', fontsize=14)
    ax[1].imshow(images[0])
    plt.tight_layout()
    plt.show()


def loss(output):
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    return output[0]


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m


LOAD_IMG = 'test4.jpg'


def output():
    model = keras.models.load_model(MODEL_NAME)

    model.summary()

    # load the image with the required shape
    img = load_img(LOAD_IMG, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)

    arr = [img]

    imgarr = np.array(arr)

    # model = Model(inputs=model.inputs, outputs=outputs)

    out = model.predict(imgarr)

    print(np.asarray(out[0]))

    out_3_point = np.reshape(np.asarray(out[0]), (3, 2))
    out_img = np.zeros([224, 224, 3], dtype=np.uint8)
    # Plot Pixel
    for i in range(0, len(out_3_point)):
        # print(round(points[i][0] * width))
        # print(round(points[i][1] * height))
        out_img[round(out_3_point[i][1]), round(out_3_point[i][0])][:] = 255
    cv2.imshow('out_img', out_img)
    cv2.imshow('original img', cv2.imread(LOAD_IMG))

    text_file = open("out.txt", "w")
    text_file.write(str(np.asarray(out[0])))
    text_file.close()

    # for pixel map
    # cv2.imshow('out', out[0])

    cv2.waitKey(10000)


IMG_NAME = 'iso2.jpg'

if __name__ == '__main__':
    # visualize()
    visualize_all()
    # output()
