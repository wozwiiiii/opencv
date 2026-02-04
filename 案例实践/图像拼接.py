import cv2
import numpy as np

# 读取两个图像
img1 = cv2.imread('road1.jpg')
img2 = cv2.imread('road2.jpg')

# 将两个图像拼接成一个图像
stitcher = cv2.Stitcher.create()
result, pano = stitcher.stitch([img1, img2])

if result == cv2.Stitcher_OK:
    cv2.imshow('Panorama', pano)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("Error during stitching.")