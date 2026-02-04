import cv2
import os
import numpy as np
import sys


print("=" * 50)
print("人脸检测程序启动")
print("=" * 50)

# ========== 1. 加载人脸检测器（使用绝对英文路径） ==========
print("\n[步骤1] 加载人脸检测模型...")

# 我们尝试的英文路径（请确保你已经把XML文件放在其中一个位置）
possible_xml_paths = [
    # 首选：C盘根目录（最可靠）
    r'C:\haarcascade_frontalface_default.xml',
    # 备用：D盘根目录
    r'D:\haarcascade_frontalface_default.xml',
    # 备用：当前用户目录（如果用户名是英文）
    os.path.join(os.path.expanduser('~'), 'haarcascade_frontalface_default.xml'),
]

face_cascade = None
used_path = None

for xml_path in possible_xml_paths:
    print(f"  尝试路径: {xml_path}")
    if os.path.exists(xml_path):
        face_cascade = cv2.CascadeClassifier(xml_path)
        if not face_cascade.empty():
            used_path = xml_path
            print(f"  ✅ 成功从以下路径加载: {xml_path}")
            break
        else:
            print(f"  ❌ 文件存在但加载失败，可能已损坏")
    else:
        print(f"  ⚠️  文件不存在")

# 如果所有路径都失败了
if face_cascade is None or face_cascade.empty():
    print("\n❌ 错误：无法加载人脸检测器！")
    print("\n请手动执行以下操作：")
    print("1. 访问以下网址下载XML文件：")
    print("   https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
    print("2. 将下载的文件保存到一个纯英文路径，例如：")
    print("   C:\\haarcascade_frontalface_default.xml")
    print("3. 然后重新运行此程序。")
    sys.exit(1)  # 退出程序


# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧
    ret, frame = cap.read()

    # 将帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用级联分类器检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 为每个检测到的人脸绘制一个矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Faces found', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()