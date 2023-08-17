import numpy as np
import cv2
cap = cv2.VideoCapture('data/MOT17-09-DPM-raw.mp4')

feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.1,
                      minDistance = 10,
                      blockSize = 10)

lk_params = dict(winSize = (10, 10), maxLevel = 1,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # create some random colors
color = np.random.randint(0, 255, (100, 3)) # take first frame and find corners in itret
# old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BAYER_BG2GRAY)
# Assuming you have already initialized the video capture object 'cap'
ret, old_frame = cap.read()
print(ret)
# Check if the frame was successfully read
if ret:
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
else:
    # Handle the case when no frame was read
    print("Error: Unable to read frame from the video capture.")

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) # create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Lucas-Kanade optical flow on video
while(cap.isOpened()):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) # select good points to predict
    good_new = p1[st==1]
    good_old = p0[st==1] # draw tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        a = int(a)
        b = int(b)
        c,d = old.ravel()
        c = int(c)
        d = int(d)
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k=cv2.waitKey(150) & 0xff
    if k == 27:
        break # now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows
cap.release()