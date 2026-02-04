import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

# 读取第一帧
ret, frame = cap.read()

# 设置初始的窗口位置
r, h, c, w = 240, 100, 400, 160
track_window = (c, r, w, h)

# 设置初始的ROI用于跟踪
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置终止条件，迭代10次或者至少移动1次
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(True):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 使用MeanShift算法找到新的位置
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # 在图像上画出新的窗口位置
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow('img2', img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()