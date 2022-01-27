from flask import Flask, request, Response
from numpy import savez_compressed

from CSL_demo import predict_ctc, generate_data, predict_ctc_custom
from urllib.parse import unquote
from flask import jsonify
import cv2
import threading
from flask_cors import CORS, cross_origin

from datetime import datetime

import tensorflow as tf
import gc
import os

import numpy as np

from extract_layer4 import get_output_layer
from keypoint_hrnet import predict_keypoint

from numba import cuda

import math

from os import listdir
from os.path import isfile, join

device = cuda.get_current_device()

PREVIEW = True
DEBUG = True

CROP_X = 200
CROP_TOP = 200

# Path Config #
# TEMP_DIR = r"F:\Dataset\Sign Language\Demo CSL\Temp\\"
# DATASET_ROOT = r"F:\Dataset\Sign Language\TSL\\"
PREDICTION_DIR = r"F:\Dataset\TSL\Data Predicted\\"
DATA_COLLECTION = r"F:\Dataset\TSL\Data Collection\\"

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

lock = threading.Lock()


@app.route('/')
def index():
    return 'Index Page'


@app.route('/predict')
@cross_origin(origin='*')
def predict():
    filepath = request.args.get('filepath')
    ground_truth, prediction, video_path = '', '', ''
    value = {}
    try:
        # ground_truth, prediction, video_path = predict_ctc(file_path=unquote(filepath).replace('\\\\', '\\'))
        ground_truth, prediction, video_path = predict_ctc(file_path=unquote(filepath))

        value = {
            "ground_truth": ground_truth,
            "prediction": prediction,
            "video_path": video_path
        }
    except ImportError:
        print('error')
    finally:
        return value


@app.route('/file-path')
def file_path():
    x_data, y_data, x_len, y_len, x_data_keypoint, classes_col, target_data, video_path = generate_data(
        class_count=10, get_frame_length=False)

    print('flask')
    print(x_data)

    value = {
        "filepath": x_data.tolist()
    }

    return jsonify(value)


def generate(filepath=r'F:\Dataset\Sign Language\CSL\pytorch\color/000005/P01_s1_05_3._color.avi'):
    # grab global references to the lock variable
    global lock
    # initialize the video stream
    vc = cv2.VideoCapture(filepath)

    # vc.set(cv2.CAP_PROP_FPS, 10)

    # check camera is open
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # while streaming
    while rval:
        # wait until the lock is acquired
        with lock:
            # read next frame
            rval, frame = vc.read()
            # if blank frame
            if frame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            cv2.waitKey(10)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    # release the camera
    vc.release()


@app.route('/stream', methods=['GET'])
def stream():
    filepath = request.args.get('filepath')
    return Response(generate(filepath), mimetype="multipart/x-mixed-replace; boundary=frame")


# Getting arguments from a POST form
@app.route("/send-video", methods=['POST'])
def send_video():
    video = request.files.get('video')

    now = datetime.now()  # current date and time

    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    print("date and time:", date_time)

    video.filename = date_time + ".mp4"

    # Saved Video Path
    filename = PREDICTION_DIR + video.filename

    print(filename)

    video.save(filename)

    video = []

    predict_keypoint(input=filename, output=filename[:-4] + '.npy')
    print("predict keypoint done.")

    if DEBUG:
        print(video)

    save_file = f'{PREDICTION_DIR}\{date_time}.npz'

    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_FPS, int(30))

    try:
        video = []

        while cap.isOpened():
            ret, frame = cap.read()
            # (height, width) = frame.shape[:2]
            if ret:

                if PREVIEW:
                    cv2.imshow('orig', frame)

                cropped = frame[0 + CROP_TOP:720, 0 + CROP_X:1280 - CROP_X]

                if PREVIEW:
                    cv2.imshow('cropped', cropped)

                resized_image = cv2.resize(cropped, (224, 224))

                if PREVIEW:
                    cv2.imshow('resized', resized_image)

                # append frame to be converted
                video.append(np.asarray(resized_image))

                if PREVIEW:
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            else:
                break

        # cv2.waitKey(125)

        if DEBUG:
            print(np.asarray(video).shape)
            print(length)

        output = get_output_layer(src=np.asarray(video))
        tf.keras.backend.clear_session()
        gc.collect()

        if DEBUG:
            print(output.shape)
            # print(output)

        # save input video
        savez_compressed(save_file, output)

        # exit(0)

        cap.release()

    except cv2.error as e:
        print(e)
        False

    length_video = np.asarray(video).shape

    print(length_video[0])
    print(length_video)

    print("output done.")

    tf.keras.backend.clear_session()
    gc.collect()

    # device.reset()
    print("reset cuda")

    # ground_truth, prediction, video_path = predict_ctc_custom(npz_path="D:\\Dataset\\Sign Language\\CSL-Output/000000/P01_s1_00_0._color.npz",
    #                    npy_path=r"F:\Dataset\Sign Language\CSL-Key\000000\P01_s1_00_0._color.avi.npy", frame_len=106)\

    ground_truth, prediction, video_path = predict_ctc_custom(npz_path=save_file, npy_path=filename[:-4] + '.npy',
                                                              frame_len=int(math.ceil(length_video[0] / 2)))

    print('PREDICTION FROM MODULE')
    print(prediction)

    prediction_str = " ".join(map(str, prediction))

    print(prediction_str)

    filename = f'{PREDICTION_DIR}\{date_time}.txt'
    if not os.path.isfile(filename):
        open(filename, 'w').close()

    with open(filename, "a", encoding='utf-8') as textfile:
        textfile.write(str(prediction_str))

    value = {
        "ok": True,
        "prediction": prediction_str,
        "fileName": date_time
    }

    return value


@app.route("/upload-video", methods=['POST'])
def upload_video():
    video = request.files.get('video')
    label = request.form.get('label').zfill(2)
    subject = request.form.get('subject').zfill(2)

    print(video)
    print(label)
    print(subject)

    # respective label directory
    label_path = DATA_COLLECTION + label

    # list all files on the label directory
    onlyfiles = [f for f in listdir(label_path) if isfile(join(label_path, f))]

    sample_number = 0

    for files in onlyfiles:
        if files[:2] == subject:
            sample_number = sample_number + 1

    print('new sample number : ' + str(sample_number))

    video.filename = "{}_{}.mp4".format(subject, str(sample_number).zfill(2))

    ## Saved Video Path
    video_path = "{}\{}".format(label_path, video.filename)
    #
    print(video_path)
    #
    video.save(video_path)

    value = {
        "ok": True,
        "saved_path": video_path,
        "label": label,
        "subject": subject
    }

    return value


@app.route("/fix-label", methods=['POST'])
def fix_label():
    label = request.form.get('fixedLabel')
    filename = request.form.get('fileName')

    print(label)
    print(filename)

    # respective label directory
    txt_path = PREDICTION_DIR + f'{filename}.txt'

    with open(txt_path, "a", encoding='utf-8') as textfile:
        textfile.write(f"\n{str(label)}")

    value = {
        "ok": True,
        "saved_path": txt_path,
    }

    return value


if __name__ == '__main__':
    # modelfile = 'models/final_prediction.pickle'
    # model = p.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')
