from werkzeug.wrappers import Request, Response
from flask import Flask, redirect, url_for, render_template, send_from_directory
from flask import request
import requests
from flask import jsonify
from flask_script import Manager
import traceback
import base64
import time
import warnings
import os
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

app = Flask(__name__)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)

        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

        # load our serialized model from disk
        print("[INFO] loading model...")
        args = {
            "prototxt": "MobileNetSSD_deploy.prototxt.txt",
            "model": "MobileNetSSD_deploy.caffemodel",
            "image": "images/"+filename,
            "confidence": 0.5
        }
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        # (note: normalization is done via the authors of the MobileNet SSD
        # implementation)
        image = cv2.imread(args["image"])
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                if CLASSES[idx] == 'person':
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print("[INFO] {}".format(label))

                    crop_img = image[startY:endY, startX:endX]
                    cv2.rectangle(image, (startX, startY), (endX, endY),
                        COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                    
                    person_path = 'person_images/'
                    if not os.path.exists(str(person_path)):
                        os.makedirs(str(person_path))
                    person_file_path = person_path+filename
                    cv2.imwrite(person_file_path, crop_img)
                    
                    return render_template("complete.html", image_name=filename, person='Human Detected')

                    
             

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename, person='No Human Detected')
@app.route('/upload/<path>/<filename>', methods=['GET', 'POST'])
def send_image(path, filename):
    print(filename)
    return send_from_directory(path, filename)

if __name__ == '__main__':
    app.run()
