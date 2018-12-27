'''Lane Detection using perspective transform. Convert frame to a bird's eye view which eliminates the surrounding objects. Then using histogram of the frame, 
    detect the base point of lanes and track and update from that point.'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

video_file = "../video 2_processed.mov"
cap = cv2.VideoCapture(video_file)

if(cap.isOpened() == False):
    print("Error opening video file")

DOWNSAMPLE_RATIO = 2
GAMMA = 1.5
invGamma = 1.0/GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")\

def gammaCorrection(image):
    ''' Handle light variability using gamma correction with gamma value of 1.5 '''
    return cv2.LUT(image, table)

def roi(image, vertices):
    ''' Crop the region of interest i.e. area containing the roads'''
    mask = np.zeros_like(image)
    # channels = image.shape[2]

    # mask_color = (255,) * channels
    mask_color = 255
    cv2.fillPoly(mask, np.array([vertices]), mask_color)
    masked_img = cv2.bitwise_and(image, mask)

    return masked_img

def isolateYellowMask(image):
    '''Get the yellow lanes in the image'''
    lower = np.array([6, 103, 27], dtype = "uint8")
    upper = np.array([45, 153, 85], dtype = "uint8")

    yellow_mask = cv2.inRange(image, lower, upper)
    return yellow_mask

def isolateWhiteMask(image):
    '''Get the white lanes in the frame'''
    lower = np.array([30, 115, 0], dtype = "uint8")
    upper = np.array([140, 200, 30], dtype = "uint8")

    white_mask = cv2.inRange(image, lower, upper)
    return white_mask

def trackLanesInitialize(hsl_mask):
    '''inital lane detection'''
    histogram = np.sum(hsl_mask[int(height/2):, :], axis=0)
    # plt.figure();
    # plt.plot(histogram);
    # out_img = np.dstack((hsl_mask, hsl_mask, hsl_mask)) * 255
    mid_point = np.int(histogram.shape[0]/2)
    leftxBase = np.argmax(histogram[:mid_point])
    rightxBase = np.argmax(histogram[mid_point:]) + mid_point

    nwindows = 9
    window_height = np.int(height/nwindows)
    nonzero = np.nonzero(hsl_mask)
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    leftxCurrent = leftxBase
    rightxCurrent = rightxBase

    margin = 100
    minPix = 50

    leftLaneInds = []
    rightLaneInds = []
    # c = 0
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(hsl_mask.shape[0] - (window+1)*window_height)
        win_y_high = int(hsl_mask.shape[0] - window*window_height)
        win_xleft_low = leftxCurrent - margin
        win_xleft_high = leftxCurrent + margin
        win_xright_low = rightxCurrent - margin
        win_xright_high = rightxCurrent + margin

        
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3)

        goodLeftInds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        goodRightInds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x <= win_xright_high)).nonzero()[0]


        leftLaneInds.append(goodLeftInds)
        rightLaneInds.append(goodRightInds)

        if len(goodLeftInds) > minPix:
            leftxCurrent = np.int(np.mean(nonzero_x[goodLeftInds]))

        if len(goodRightInds) > minPix:
            rightxCurrent = np.int(np.mean(nonzero_x[goodRightInds]))

    newleftLaneInds = np.concatenate(leftLaneInds)
    newrightLaneInds = np.concatenate(rightLaneInds)

    leftx = nonzero_x[newleftLaneInds]
    lefty = nonzero_y[newleftLaneInds]
    rightx = nonzero_x[newrightLaneInds]
    righty = nonzero_y[newrightLaneInds]

    leftFit = np.polyfit(lefty, leftx, 2)
    rightFit = np.polyfit(righty, rightx, 2)

    # #####################################################
    # ploty = np.linspace(0, hsl_mask.shape[0] - 1, hsl_mask.shape[0])
    # leftFitX = leftFit[0] * ploty**2 + leftFit[1] * ploty + leftFit[2]
    # rightFitX = rightFit[0] * ploty**2 + rightFit[1] * ploty + rightFit[2]

    # nonzero = np.nonzero(hsl_mask)
    # nonzero_x = np.array(nonzero[1])
    # nonzero_y = np.array(nonzero[0])

    # margin = 100

    # leftLaneInds = ((nonzero_x > (leftFit[0] * nonzero_y**2 + leftFit[1] * nonzero_y + leftFit[2] - margin)) & 
    #                 (nonzero_x < (leftFit[0] * nonzero_y**2 + leftFit[1] * nonzero_y + leftFit[2] + margin)))

    # rightLaneInds = ((nonzero_x > (rightFit[0] * nonzero_y**2 + rightFit[1] * nonzero_y + rightFit[2] - margin)) & 
    #                 (nonzero_x < (rightFit[0] * nonzero_y**2 + rightFit[1] * nonzero_y + rightFit[2] + margin)))

    # leftx = nonzero_x[leftLaneInds]
    # lefty = nonzero_y[leftLaneInds]
    # rightx = nonzero_x[rightLaneInds]
    # righty = nonzero_y[rightLaneInds]

    # leftFit = np.polyfit(lefty, leftx, 2)
    # rightFit = np.polyfit(righty, rightx, 2)

    return leftFit, rightFit

def trackLanesUpdate(hsl_mask, leftFit, rightFit):
    '''update the lanes instead of detecting in each frame'''
    global window_search
    global frame_count

    if frame_count%20 == 0:
        window_search = True


    nonzero = np.nonzero(hsl_mask)
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    margin = 100

    leftLaneInds = ((nonzero_x > (leftFit[0] * nonzero_y**2 + leftFit[1] * nonzero_y + leftFit[2] - margin)) & 
                    (nonzero_x < (leftFit[0] * nonzero_y**2 + leftFit[1] * nonzero_y + leftFit[2] + margin)))

    rightLaneInds = ((nonzero_x > (rightFit[0] * nonzero_y**2 + rightFit[1] * nonzero_y + rightFit[2] - margin)) & 
                    (nonzero_x < (rightFit[0] * nonzero_y**2 + rightFit[1] * nonzero_y + rightFit[2] + margin)))

    leftx = nonzero_x[leftLaneInds]
    lefty = nonzero_y[leftLaneInds]
    rightx = nonzero_x[rightLaneInds]
    righty = nonzero_y[rightLaneInds]

    leftFit = np.polyfit(lefty, leftx, 2)
    rightFit = np.polyfit(righty, rightx, 2)

    # ploty = np.linspace(0, hsl_mask.shape[0] - 1, hsl_mask.shape[0])
    # leftFitX = leftFit[0] * ploty**2 + leftFit[1] * ploty + leftFit[2]
    # rightFitX = rightFit[0] * ploty**2 + rightFit[1] * ploty + rightFit[2]

    return leftFit, rightFit #, leftx, lefty, rightx, righty

def drawLane(hsl_mask, imSmall, leftFit, rightFit):
    ploty = np.linspace(0, hsl_mask.shape[0] - 1, hsl_mask.shape[0])
    leftFitX = leftFit[0] * ploty**2 + leftFit[1] * ploty + leftFit[2]
    rightFitX = rightFit[0] * ploty**2 + rightFit[1] * ploty + rightFit[2]

    warp_zero = np.zeros_like(hsl_mask).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([leftFitX, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightFitX, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, inverse_perspective_transform, (hsl_mask.shape[1], hsl_mask.shape[0]))

    res = cv2.addWeighted(imSmall, 1, newwarp, 0.5, 0.0)
    return res

def initParams(imSmall):
    '''define parameters'''
    height, width = imSmall.shape[:2]
    vertices = [(0, height), (width//2 + 50, height//2 + 10), (width, height),] #(0, height//2),

    source_points = np.float32([
                        [557, 360],
                        [652, 360],
                        [837, 540], 
                        [309, 540]])

    destination_points = np.float32([
                                    [306, 0],
                                    [712, 0],
                                    [712, imSmall.shape[0]],
                                    [306, imSmall.shape[0]]])

    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)

    return height, width, perspective_transform, inverse_perspective_transform

def getBinaryMask(imSmall):
    '''pipeline for pre processing the frame'''
    gamma = gammaCorrection(imSmall)

    # cropped_img = roi(gamma, vertices)
    warped_img = cv2.warpPerspective(gamma, perspective_transform, (width, height), flags=cv2.INTER_LINEAR)
    hsl = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HLS)
    yellow_mask = isolateYellowMask(hsl)
    white_mask = isolateWhiteMask(hsl)
    hsl_mask = cv2.bitwise_or(yellow_mask, white_mask)

    return hsl_mask

first_Frame = True

global window_search 
global frame_count
window_search = True
frame_count = 0
total_time = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    t = time.time()

    if ret == False:
        print("Not able to read video stream")

    imSmall = cv2.resize(frame, None, 
                        fx = 1.0/DOWNSAMPLE_RATIO, 
                        fy = 1.0/DOWNSAMPLE_RATIO, 
                        interpolation = cv2.INTER_LINEAR)

    if first_Frame:
        height, width, perspective_transform, inverse_perspective_transform = initParams(imSmall)
        vid_writer = cv2.VideoWriter('perspective_lane_tracking.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width,height))
        first_Frame = False
    
    hsl_mask = getBinaryMask(imSmall)

    if window_search:
        leftFit, rightFit = trackLanesInitialize(hsl_mask)        

    else:
        leftFit = leftFitPrev
        rightFit = rightFitPrev
        leftFit, rightFit = trackLanesUpdate(hsl_mask, leftFit, rightFit)

    leftFitPrev = leftFit
    rightFitPrev = rightFit
    result = drawLane(hsl_mask, imSmall, leftFit, rightFit)

    t2 = time.time() - t

    total_time += t2
    fps = frame_count/total_time

    cv2.putText(result, "FPS: {}".format(round(fps, 3)), (result.shape[0] // 2, result.shape[1]//2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("test", result)
    vid_writer.write(result)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
vid_writer.release()
cv2.destroyAllWindows()