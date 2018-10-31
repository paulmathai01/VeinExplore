import argparse

import cv2
import numpy as np
from imutils.video import FPS
# import Image
from skimage import img_as_bool
from skimage import img_as_ubyte
from skimage.morphology import skeletonize

IMG_HEIGHT = 360
IMG_WIDTH = 720


def adjust_sharpness(imgIn):
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    kernel = kernel - boxFilter
    custom = cv2.filter2D(imgIn, -1, kernel)
    return (custom)

"""
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to video", )
args = vars(ap.parse_args())
"""

fgbg = cv2.createBackgroundSubtractorMOG2()

# vs = cv2.VideoCapture(args["video"])

vs = cv2.VideoCapture("http://192.168.43.253:8000/eyel.mjpeg")


a = 0
while a != -1:
    a = a + 1
    (grabbed, frame) = vs.read()
    if not grabbed:
        a = -1
        break
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    M = np.ones(frame.shape, dtype="uint8") * 50
    added = cv2.add(frame, M)
    img = cv2.cvtColor(added, cv2.COLOR_BGR2GRAY)
    th, img = cv2.threshold(img, 100, 256, cv2.THRESH_TOZERO)
    print("[INFO] Converted from RGB to GRAY for interation=", a)
    img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = adjust_sharpness(img)
    # img = adjust_sharpness(img)

    img = cv2.adaptiveThreshold(img, 4000, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 47, 2)
    img = cv2.resize(img, (1060, 600))
    cv2.imshow("Frame", img)
    img = cv2.resize(img, (720, 360))
    kernel = np.ones((4, 4), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = img_as_bool(img)
    img = skeletonize(img)
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(frame, 1, img, 1, 1)
    img = cv2.resize(img, (1060, 600))
    print(img)
    cv2.imshow("OG", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
