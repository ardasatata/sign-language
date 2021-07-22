# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import glob

from os import listdir
from os.path import isfile, join

import cv2

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

mp_holistic = mp.solutions.holistic

# DIR = r"G:\TSL\Dataset-Test"
DIR = r"D:\CSL-25K\pytorch\color\000009"

DIR2 = r"G:\TSL\temp\img\\"

DIR3 = r"G:\TSL\temp\annotation\\"

SAVE_PATH = r'G:\TSL\Color-Cropped\000009\\'


def print_hi(name):
    folders = [f.path for f in os.scandir(DIR) if f.is_dir()]
    print(folders)

    onlyfiles = [f for f in listdir(DIR) if isfile(join(DIR, f))]

    print(onlyfiles)

    filePaths = glob.glob(DIR + "/*")

    count = 0

    for file in filePaths:
        try:
            cap = cv2.VideoCapture(file)

            cap.set(1, 30)
            ret, frame = cap.read()
            cv2.imwrite(DIR2 + str(count) + '.jpg', frame)
            filename = DIR2 + str(count) + '.jpg'

            print(file[32:-4])

            nose_x, nose_y = get_position(filename, str(count))
            #
            crop_video(file, file[32:-4] + '.avi', nose_x, nose_y)

            cap.release()
            count += 1

        except cv2.error as e:
            print(e)
            False


def get_position(file, filename):
    # For static images:
    pose = mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.4)

    image = cv2.imread(file)
    image_hight, image_width, _ = image.shape

    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # if not results.pose_landmarks:
    #     continue

    # print(results.pose_landmarks)

    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
    )
    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # cv2.imwrite(DIR3 + 'annotated_image_' + filename + '.png', annotated_image)
    pose.close()

    nose_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width
    nose_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight

    return nose_x, nose_y


def crop_video(file, filename='test1.avi', x=640, y=360):
    cap = cv2.VideoCapture(file)
    cap.set(1, 2)

    x1, x2, y1, y2, width, height = crop_location(x, y)

    outcrop = cv2.VideoWriter(SAVE_PATH + filename, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))

    # print(width, height)
    print(file)

    while cap.isOpened():
        ret, frame = cap.read()
        # (height, width) = frame.shape[:2]
        if ret:
            # Y , X
            cropped = frame[y1:y2, x1:x2]

            outcrop.write(cropped)
            # cv2.imshow('vid', cropped)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    outcrop.release()
    cv2.destroyAllWindows()


def crop_location(x_pos, y_pos):
    X1 = 300
    Y1 = 120
    Y2 = 420

    XPOS = int(x_pos)
    YPOS = int(y_pos)

    x1 = XPOS - X1
    x2 = XPOS + X1
    y1 = YPOS - Y1
    y2 = YPOS + Y2

    return x1, x2, 120, 720, x2 - x1, 600



def test_get_position(file):
    cap = cv2.VideoCapture(file)

    cap.set(1, 10)
    ret, frame = cap.read()
    cv2.imwrite(DIR2 + 'test' + '.jpg', frame)
    filename = DIR2 + 'test' + '.jpg'

    print(file[17:28])

    # exit(0)

    nose_x, nose_y = get_position(filename, 'test')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # test_get_position(r'G:\TSL\Dataset-2\P09_s1_01_1._color.avi')
    # get_position(r'G:\TSL\Dataset-2\P09_s1_01_1._color.avi', 'file0.jpg')
    # crop_video(r'G:\TSL\Dataset-Test\P01_s1_00_0._color.avi', 'test22.avi')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
