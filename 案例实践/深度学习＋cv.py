import cv2
import numpy as np

# 加载预训练的模型
net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')
# 加载预训练的模型
net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')

# 加载标签名
with open('synset_words.txt', 'r') as f:
    labels = f.read().strip().split("\n")

# 加载图像，并进行预处理
image = cv2.imread('image.jpg')
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# 将图像输入到网络中，进行前向传播，得到输出结果
net.setInput(blob)
outputs = net.forward()

# 获取预测结果
class_id = np.argmax(outputs)
label = labels[class_id]

print('Output class:', label)