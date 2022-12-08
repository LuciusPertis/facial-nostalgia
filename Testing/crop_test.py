import cv2 as cv
import video

cap = video.create_capture()
while True:
    ret, img = cap.read()
    cv.imshow('capture', img)
    ch = cv.waitKey(1)
    if ch == 27:
        break

fname = "./simple_face.jpg"

fimg = cv.imread(fname)

cv.imshow("t0", fimg)