import numpy as np
import argparse
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-i2", "--imagetwo", required = False, help = "Path to logo", default = "twitter-logo.png")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
logo = cv2.imread(args["imagetwo"])
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5,5), 0)
cv2.imshow("Image", image)
cv2.waitKey(0)

# -----------------------------------------------
# Face Detection using DNN Net
# -----------------------------------------------
# detect faces using a DNN model 
# download model and prototxt from https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models

def detectFaceOpenCVDnn(net, frame):
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(1):
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        bboxes.append([x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        
        top=x1
        right=y1
        bottom=x2-x1
        left=y2-y1

        face = frame[right:right+left, top:top+bottom]
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

logoMask = cv2.inRange(logo, 125, 255)
print(logoMask.shape)
cv2.imshow("mask", logoMask)
cv2.waitKey(0)

height, width = logoMask.shape
blank = np.zeros((height, width), dtype = "uint8")
newMask = logoMask + blank

slicedFace = cv2.resize(face, (width, height), interpolation = cv2.INTER_AREA)
cv2.imshow("newFace", slicedFace)

masked = cv2.bitwise_and(slicedFace, slicedFace, mask = logoMask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)