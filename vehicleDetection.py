import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import scipy.io
import time

# from keras.models import Sequential, Model
# from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
# from keras.layers.advanced_activations import LeakyReLU
# from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# from keras.optimizers import SGD, Adam
# from keras.models import load_model, Model

DOWNSAMPLE_RATIO = 2
CONFIDENCE = 0.4
THRESH = 0.5
inpWidth = 416
inpHeight = 416

GAMMA = 1.5
invGamma = 1.0/GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")\

def gammaCorrection(image):
    return cv2.LUT(image, table)

def drawBBoxes(classID, conf, left, top, right, bottom):
    color = [int(c) for c in COLORS[classID]]
    cv2.rectangle(frame, (left, top), (right, bottom), color)

    text = "{} : {:.4f}".format(LABELS[classID], conf)

    labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def postProcess(frame, outs):
    frameHeight, frameWidth = frame.shape[:2]

    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                (centerX, centerY, width, height) = box.astype("int")

                left = int(centerX - (width / 2))
                top = int(centerY - (height / 2))

                boxes.append([left, top, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESH)

    if len(idxs) > 0:
        for i in idxs:
            i = i[0]
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            

            drawBBoxes(classIDs[i], confidences[i], x, y, x + w, y + h)



yolo_model_path = "models/yolov3-tiny.h5"
yolo_weights_path = "models/yolov3-tiny.weights"
yolo_cfg_path = "models/yolov3-tiny.cfg"
yolo_labels_path = "models/coco.names"

with open(yolo_labels_path, 'rt') as f:
    LABELS = f.read().rstrip("\n").split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(yolo_cfg_path, yolo_weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

video_file = "../video 2_processed.mov"
cap = cv2.VideoCapture(video_file)

if(cap.isOpened() == False):
    print("Error opening video file")

firstFrame = True

while(cap.isOpened()):

    ret, frame = cap.read()
    if ret == False:
        print("Not able to read video stream")

    frame = cv2.resize(frame, None, 
                    fx = 1.0/DOWNSAMPLE_RATIO, 
                    fy = 1.0/DOWNSAMPLE_RATIO, 
                    interpolation = cv2.INTER_LINEAR)

    if firstFrame:
        height, width = frame.shape[:2]
        vid_writer = cv2.VideoWriter('vehicle_detection_yolo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width,height))
        firstFrame = False

    frame = gammaCorrection(frame)

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (inpWidth, inpHeight), swapRB = True, crop=False)
    
    net.setInput(blob)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    layerOutputs = net.forward(ln)

    postProcess(frame, layerOutputs)

    t, _ = net.getPerfProfile()
    label = 'FPS: %.2f' % (cv2.getTickFrequency() / t)
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    cv2.imshow("test", frame)
    vid_writer.write(frame)
    # cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
vid_writer.release()
cv2.destroyAllWindows()

# NORM_H, NORM_W = 416, 416
# GRID_H, GRID_W = 13 , 13
# BATCH_SIZE = 8
# BOX = 5
# ORIG_CLASS = 20

# model = Sequential()

# #Layer 1
# model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(416,416,3)))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Layer 2 - 5
# for i in range(0, 4):
#     model.add(Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

# # Layer 6
# model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))

# # Later 7
# model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.1))

# # Later 8
# model.add(Conv2D(256, (1,1), strides=(1,1), padding='same', use_bias=False))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.1))

# # Layer 9
# model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.1))

# # Layer 10
# model.add(Conv2D(BOX * (4 + 1 + ORIG_CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))


# model.add(Activation('linear'))
# model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + ORIG_CLASS)))

# model.summary()