import cv2
import numpy as np
from scipy import stats

DOWNSAMPLE_RATIO = 2
GAMMA = 1.5
invGamma = 1.0/GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")\

def gammaCorrection(image):
    return cv2.LUT(image, table)

def drawLines(image, lines, color=(0, 0, 255), thickness=3):
    if lines is None:
        return image

    img = np.copy(image)
    # line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # img = cv2.addWeighted(image, 0.8, line_img, 1.0, 0.0)
    return img

def roi(image, vertices):
    mask = np.zeros_like(image)
    # channels = image.shape[2]

    # mask_color = (255,) * channels
    mask_color = 255
    cv2.fillPoly(mask, np.array([vertices]), mask_color)
    masked_img = cv2.bitwise_and(image, mask)

    return masked_img

def isolateYellowMask(image):
    lower = np.array([6, 103, 27], dtype = "uint8")
    upper = np.array([45, 153, 85], dtype = "uint8")

    yellow_mask = cv2.inRange(image, lower, upper)
    return yellow_mask

def isolateWhiteMask(image):
    lower = np.array([30, 115, 0], dtype = "uint8")
    upper = np.array([140, 200, 30], dtype = "uint8")

    white_mask = cv2.inRange(image, lower, upper)
    return white_mask

def seperateLaneLines(lines):
    left_lane = []
    right_lane = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1
            if dx == 0:
                continue

            dy = y2 - y1
            if dy == 0:
                continue

            slope = dy/dx
            if abs(slope) <= 0.1:
                continue

            if slope < 0:
                left_lane.append([[x1, y1, x2, y2]])

            else:
                right_lane.append([[x1, y1, x2, y2]])

    return left_lane, right_lane


def laneRegression(lines):
    xs = []
    ys = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)

    slope, intercept, _, _, _ = stats.linregress(xs, ys)

    return slope, intercept

def trackLaneLine(image, lines):
    M, c = laneRegression(lines)
    #y = Mx + c
    #x = (y - c) / M

    bottom_y = image.shape[0]
    bottom_x = (bottom_y - c)/M

    top_y = image.shape[0]/2
    top_x = (top_y - c)/M

    lines = [[[int(bottom_x), int(top_x), int(bottom_y), int(top_y),]]]

    return drawLines(image, lines)

video_file = "../video 2_processed.mov"
cap = cv2.VideoCapture(video_file)

if(cap.isOpened() == False):
    print("Error opening video file")

first_Frame = True

# def callback(x):
#     pass

# cv2.namedWindow('image')
# ilowH = 0
# ihighH = 255

# ilowL = 0
# ihighL = 255
# ilowS = 0
# ihighS = 255

# cv2.createTrackbar('lowH','image',ilowH,255,callback)
# cv2.createTrackbar('highH','image',ihighH,255,callback)

# cv2.createTrackbar('lowL','image',ilowL,255,callback)
# cv2.createTrackbar('highL','image',ihighL,255,callback)

# cv2.createTrackbar('lowS','image',ilowL,255,callback)
# cv2.createTrackbar('highS','image',ihighL,255,callback)

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    count += 1

    if ret == False:
        print("Not able to read video stream")

    imSmall = cv2.resize(frame, None, 
                        fx = 1.0/DOWNSAMPLE_RATIO, 
                        fy = 1.0/DOWNSAMPLE_RATIO, 
                        interpolation = cv2.INTER_LINEAR)

    gamma = gammaCorrection(imSmall)

    if first_Frame:
        height, width = imSmall.shape[:2]
        vertices = [(0, height), (width//2 + 50, height//2 + 10), (width, height),] #(0, height//2),
        vid_writer = cv2.VideoWriter('hough_lines_lane_tracking.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width,height))
        first_Frame = False

    hsl = cv2.cvtColor(gamma, cv2.COLOR_BGR2HLS)
    yellow_mask = isolateYellowMask(hsl)
    white_mask = isolateWhiteMask(hsl)
    hsl_mask = cv2.bitwise_or(yellow_mask, white_mask)
    combined = cv2.bitwise_and(imSmall, imSmall, mask = hsl_mask)

    gray_img = cv2.cvtColor(combined, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray_img, (5, 5), 0)

    canny_img = cv2.Canny(smooth, 50, 150)
    cropped_img = roi(canny_img, vertices)

    # ilowH = cv2.getTrackbarPos('lowH', 'image')
    # ihighH = cv2.getTrackbarPos('highH', 'image')
    # ilowL = cv2.getTrackbarPos('lowL', 'image')
    # ihighL = cv2.getTrackbarPos('highL', 'image')
    # ilowS = cv2.getTrackbarPos('lowS', 'image')
    # ihighS = cv2.getTrackbarPos('highS', 'image')

    # hsv = cv2.cvtColor(imSmall, cv2.COLOR_BGR2HSV)
    # lower_hsl = np.array([ilowH, ilowL, ilowS])
    # higher_hsl = np.array([ihighH, ihighL, ihighS])
    # mask_hsl = cv2.inRange(hsl, lower_hsl, higher_hsl)

    lines = cv2.HoughLinesP(cropped_img, 
                            rho = 1, 
                            theta = np.pi/45,
                            threshold = 50, 
                            lines = np.array([]),
                            minLineLength = 30,
                            maxLineGap = 10)

    if lines is not None:
        left_lane, right_lane = seperateLaneLines(lines)
        leftLaneImg = drawLines(imSmall, left_lane, color = (255, 0, 0))
        rightLaneImg = drawLines(leftLaneImg, right_lane)

        # leftRegressImg = trackLaneLine(rightLaneImg, left_lane)
        # rightRegressImg = trackLaneLine(rightLaneImg, right_lane)

    # if count == 100:
    #     print("saving image")
    #     cv2.imwrite("test.jpg", rightLaneImg)

    cv2.imshow("test", rightLaneImg)
    vid_writer.write(rightLaneImg)
    # cv2.imshow("test1", mask_hsl)
    # cv2.imshow("test2", imSmall)
    # cv2.imshow("image")


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
vid_writer.release()
cv2.destroyAllWindows()