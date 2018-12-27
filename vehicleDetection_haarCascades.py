import cv2
import numpy as np
import time
from nms import non_max_suppression_fast
DOWNSAMPLE_RATIO = 2
THRESH = 0.5
GAMMA = 1.5

invGamma = 1.0/GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

def gammaCorrection(image):
    return cv2.LUT(image, table)

cascade = cv2.CascadeClassifier("models/cars.xml")
video_file = "../video 2_processed.mov"
cap = cv2.VideoCapture(video_file)

if(cap.isOpened() == False):
    print("Error opening video file")


firstFrame = True

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        print("Not able to read video stream")
        break

    imSmall = cv2.resize(frame, None, 
                        fx = 1.0/DOWNSAMPLE_RATIO, 
                        fy = 1.0/DOWNSAMPLE_RATIO, 
                        interpolation = cv2.INTER_LINEAR)

    gamma = gammaCorrection(imSmall)

    if firstFrame:
        height, width = imSmall.shape[:2]
        vid_writer = cv2.VideoWriter('vehicle_detection_haar_cascades.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width,height))
        firstFrame = False

    gray = cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY)
    bboxes = cascade.detectMultiScale(gray, 1.2, 5)
    sup_bboxes = non_max_suppression_fast(bboxes, THRESH)

    if len(sup_bboxes) > 0:
        car_count = 0
        for (x, y, w, h) in sup_bboxes:
            if y > height/2:
                cv2.rectangle(imSmall, (x, y), (x + w, y + h), (255, 0, 0), 2)
                car_count += 1

    # result = cv2.resize(gamma, None, 
    #                     fx = 1.0/DOWNSAMPLE_RATIO, 
    #                     fy = 1.0/DOWNSAMPLE_RATIO, 
    #                     interpolation = cv2.INTER_LINEAR)

    cv2.imshow("test", imSmall)
    vid_writer.write(imSmall)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
vid_writer.release()
cv2.destroyAllWindows()
