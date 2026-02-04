import cv2
import numpy as np

# 读取图像
img = cv2.imread('road.jpg', 0)

# 使用Canny算法进行边缘检测
edges = cv2.Canny(img, 50, 150)

# 显示原图和边缘检测结果
cv2.imshow('Original Image', img)
cv2.imshow('Edge Image', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()