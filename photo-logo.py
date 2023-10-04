import numpy as np
import argparse
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
twitter = cv2.imread("twitter-logo.png")
twitter = cv2.cvtColor(twitter, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(image, (5,5), 0)
#cv2.imshow("Image", image)
#cv2.waitKey(0)

# -----------------------------------------------
# Face Detection using DNN Net
# -----------------------------------------------
# detect faces using a DNN model 
# download model and prototxt from https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models

def detectFaceOpenCVDnn(net, frame, conf_threshold=0.7):
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False,)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8,)
            
            top=x1
            right=y1
            bottom=x2-x1
            left=y2-y1

            #  blurry rectangle to the detected face
            face = frame[right:right+left, top:top+bottom]
            face = cv2.GaussianBlur(face,(23, 23), 30)
            frame[right:right+face.shape[0], top:top+face.shape[1]] = face

    return face, bboxes

# load face detection model
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

face, bboxes = detectFaceOpenCVDnn(net, image)

cv2.destroyAllWindows()
cv2.imshow("DETECTED", face)

twittermask = cv2.inRange(twitter, 125, 255)
cv2.imshow("mask", twittermask)
cv2.waitKey(0)

blank = np.zeros((244, 300), dtype = "uint8")
newMask = twittermask + blank

slicedFace = cv2.resize(face, (300, 244), interpolation = cv2.INTER_AREA)
cv2.imshow("newFace", slicedFace)

masked = cv2.bitwise_and(slicedFace, slicedFace, mask = twittermask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)