import numpy as np
import cv2 as cv

video_capture = cv.VideoCapture(cv.samples.findFile("data/MOT17-09-DPM-raw.mp4"))
ret, first_frame = video_capture.read()
previous_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255

while True:
    ret, current_frame = video_capture.read()
    if not ret:
        print("No frames grabbed!")
        break

    next_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(
        previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow("current_frame", bgr)

    k = cv.waitKey(30) & 0xFF

    if k == 27:
        break
    elif k == ord("s"):
        cv.imwrite("opticalfb.png", current_frame)
        cv.imwrite("opticalhsv.png", bgr)
    previous_frame = next_frame
    
cv.destroyAllWindows()
