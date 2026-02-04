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

# ========== 2. 准备测试图片 ==========
print("\n[步骤2] 准备测试图片...")

# 先检查是否有现成的测试图片（英文名）
test_image_found = None
image_names_to_try = [ 'face.jpg']

for img_name in image_names_to_try:
    if os.path.exists(img_name):
        test_image_found = cv2.imread(img_name)
        if test_image_found is not None:
            print(f"  ✅ 找到并加载测试图片: {img_name}")
            break

# 如果没有找到图片文件，程序会自己生成一个带“人脸”的测试图
if test_image_found is None:
    print("  ⚠️  未找到测试图片文件，正在生成模拟图像...")
    # 创建一个400x400的彩色图像作为画布
    height, width = 400, 400
    test_image_found = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 画一张简单的“脸”
    center_x, center_y = width // 2, height // 2
    face_radius = 80
    
    # 画脸（肤色椭圆）
    cv2.ellipse(test_image_found, (center_x, center_y), (face_radius, int(face_radius*1.2)), 0, 0, 360, (200, 180, 140), -1)
    
    # 画左眼和右眼
    eye_radius = 10
    cv2.circle(test_image_found, (center_x - 30, center_y - 20), eye_radius, (0, 0, 0), -1)
    cv2.circle(test_image_found, (center_x + 30, center_y - 20), eye_radius, (0, 0, 0), -1)
    
    # 画嘴巴
    cv2.ellipse(test_image_found, (center_x, center_y + 30), (40, 20), 0, 0, 180, (0, 0, 0), 3)
    
    # 画鼻子
    cv2.line(test_image_found, (center_x, center_y), (center_x, center_y + 15), (0, 0, 0), 3)
    
    print("  ✅ 模拟测试图像生成完成")

print(f"  图像尺寸: {test_image_found.shape}")

# ========== 3. 执行人脸检测 ==========
print("\n[步骤3] 执行人脸检测...")

# 转换为灰度图（人脸检测通常需要）
gray_image = cv2.cvtColor(test_image_found, cv2.COLOR_BGR2GRAY)

# 调整检测参数以获得更好效果
faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.05,    # 每次图像缩放的比例（越小越慢但越准确）
    minNeighbors=5,      # 每个候选矩形应保留的邻居个数
    minSize=(50, 50),    # 最小人脸尺寸
    maxSize=(300, 300)   # 最大人脸尺寸
)

num_faces = len(faces)
print(f"  🔍 检测到 {num_faces} 个人脸区域")

# ========== 4. 标记并显示结果 ==========
print("\n[步骤4] 标记检测结果...")

# 创建一份原图的副本用于绘制结果
result_image = test_image_found.copy()

if num_faces > 0:
    for i, (x, y, width, height) in enumerate(faces):
        # 绘制绿色矩形框标记人脸
        box_color = (0, 255, 0)  # BGR颜色：绿色
        box_thickness = 2
        cv2.rectangle(result_image, (x, y), (x + width, y + height), box_color, box_thickness)
        
        # 在人脸框上方添加标签
        label = f"Person {i+1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        label_thickness = 2
        
        # 计算文字大小以便放置背景
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
        
        # 绘制文字背景
        cv2.rectangle(result_image, 
                     (x, y - text_height - 10), 
                     (x + text_width, y), 
                     box_color, 
                     -1)  # -1 表示填充
        
        # 绘制文字
        cv2.putText(result_image, label, (x, y - 5),
                   font, font_scale, (0, 0, 0), label_thickness)  # 黑色文字
        
        print(f"    人脸 {i+1}: 位置({x}, {y}), 大小 {width}x{height}")
else:
    print("  ⚠️  未检测到人脸，这可能是由于：")
    print("     - 图片中确实没有人脸")
    print("     - 人脸太小或太大")
    print("     - 光线条件不佳")
    print("     - 人脸角度不正对摄像头")

# ========== 5. 显示和保存结果 ==========
print("\n[步骤5] 显示检测结果...")
print("  按窗口上的任意键继续...")

# 显示原图和结果对比
cv2.imshow('1. Original Image 原图', test_image_found)
cv2.imshow('2. Face Detection Result 人脸检测结果', result_image)

# 尝试保存结果到桌面（英文路径）
try:
    # 获取桌面路径（通常是英文的）
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_path = os.path.join(desktop_path, 'face_detection_result.jpg')
    
    # 如果桌面路径包含非英文字符，则保存到当前目录
    try:
        output_path.encode('ascii')
    except UnicodeEncodeError:
        desktop_path = os.getcwd()
        output_path = os.path.join(desktop_path, 'face_detection_result.jpg')
    
    cv2.imwrite(output_path, result_image)
    print(f"  💾 结果已保存至: {output_path}")
except Exception as save_error:
    print(f"  ❌ 保存结果时出错: {save_error}")
    # 尝试保存到当前目录
    cv2.imwrite('local_result.jpg', result_image)
    print("  💾 结果已保存至当前目录: local_result.jpg")

# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("程序执行完毕！")
print("=" * 50)