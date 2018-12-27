import cv2
import numpy as np


def draw_circle(event,x,y,flags,param):
    # global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        # mouseX,mouseY = x,y
        print(x, y)

img = cv2.imread("test.jpg", 1)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

# while True:
height, width = img.shape[:2]
vertices = [(0, height), (width//2 + 50, height//2 + 10), (width, height),] #(0, height//2),

source_points = np.float32([
                    [557, 360],
                    [652, 360],
                    [837, 540], 
                    [309, 540]])

destination_points = np.float32([
                                [306, 0],
                                [712, 0],
                                [712, img.shape[0]],
                                [306, img.shape[0]]])

perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)

warped_img = cv2.warpPerspective(img, perspective_transform, (width, height), flags=cv2.INTER_LINEAR)

cv2.imshow("image", warped_img)
# k = cv2.waitKey(20) & 0xFF
# if k == 27:
#     break
    # elif k == ord('a'):
    #     print(mouseX,mouseY)
cv2.waitKey(0)
cv2.destroyAllWindows()