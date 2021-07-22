import io
import sys

from PIL import Image
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from train import PoseLandmark

REF_IMG = r"C:\Users\minelab\dev\TSL\P11_s1_00_0.avi_snapshot_00.00.000.jpg"

IMG_WIDTH = 100
IMG_HEIGHT = 100
KEYPOINT = [[0.500307559967041, 0.33949682116508484],
            [0.5131224989891052, 0.31448695063591003],
            [0.5232263803482056, 0.314212441444397],
            [0.5332431197166443, 0.31403017044067383],
            [0.4846648871898651, 0.3143192231655121],
            [0.47416114807128906, 0.314197301864624],
            [0.4639100134372711, 0.31413012742996216],
            [0.5429486632347107, 0.32251256704330444],
            [0.4453672170639038, 0.32331573963165283],
            [0.517180323600769, 0.36478132009506226],
            [0.4788956046104431, 0.3646121025085449],
            [0.6055067777633667, 0.47052282094955444],
            [0.36767107248306274, 0.4661596417427063],
            [0.6543206572532654, 0.6494645476341248],
            [0.308016836643219, 0.6492811441421509],
            [0.6609691381454468, 0.8170877695083618],
            [0.2784111201763153, 0.8224952816963196],
            [0.6701566576957703, 0.8733437061309814],
            [0.25949233770370483, 0.8761253356933594],
            [0.6449164748191833, 0.8761612176895142],
            [0.2845262289047241, 0.8805792331695557],
            [0.6284334063529968, 0.8562918901443481],
            [0.3037729859352112, 0.8628993034362793],
            [0.5595301389694214, 0.8197789192199707],
            [0.41105225682258606, 0.8177140951156616]]


def generate(points, width, height, img_ref=None, write_image=False, normalized=True, full_keypoint=False):
    data = np.zeros([height, width, 3], dtype=np.uint8)

    # Plot Pixel
    for i in range(0, len(points)):
        if full_keypoint:
            data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255
        else:
            if i == PoseLandmark.LEFT_WRIST:
                data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

            if i == PoseLandmark.RIGHT_WRIST:
                data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

            if i == PoseLandmark.NOSE:
                data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

            if i == PoseLandmark.LEFT_SHOULDER:
                data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

            if i == PoseLandmark.RIGHT_SHOULDER:
                data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

            if i == PoseLandmark.LEFT_ELBOW:
                data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

            if i == PoseLandmark.RIGHT_ELBOW:
                data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

            # if i == PoseLandmark.LEFT_HIP:
            #     data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255
            #
            # if i == PoseLandmark.RIGHT_HIP:
            #     data[round(points[i][1] * height), round(points[i][0] * width)][:] = 255

    if normalized:
        normalized_pixel = np.array(NormalizeData(data), dtype='uint8')
    else:
        normalized_pixel = data

    if write_image:
        # Create a figure. Equal aspect so circles look circular
        fig, ax = plt.subplots(1, 2)

        # Show reference image
        if img_ref is not None:
            ref = plt.imread(img_ref)
            ax[1].set_title('Reference Image')
            ax[1].imshow(ref)

        ax[0].set_title('Pixel')
        ax[0].imshow(data)
        cv2.imshow('img original', data)
        cv2.imshow('normalized_pixel', normalized_pixel)
        print(data)
        print(normalized_pixel)
        print(normalized_pixel.shape)
        cv2.imwrite('result_pixel.png', data)
        cv2.imwrite('result_pixel_normalized.png', normalized_pixel)
        cv2.waitKey(10000)  # wait for 50ms
        plt.show()

    return normalized_pixel


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    generate(KEYPOINT, IMG_WIDTH, IMG_HEIGHT, REF_IMG, write_image=True)

    print(generate(KEYPOINT, 100, 100, write_image=False))
    # generate(KEYPOINT, IMG_WIDTH, IMG_HEIGHT)
