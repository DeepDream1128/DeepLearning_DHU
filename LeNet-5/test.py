import cv2
import numpy as np
import tensorflow as tf

# 加载 SavedModel
model = tf.saved_model.load('C:/Users/Even/Desktop/Lu/guidebook/minst/50epochs_train')

# 定义图像预处理函数
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(image, (28, 28))  # 调整为模型的输入大小
    img = img.reshape(-1, 28, 28, 1) / 255.0  # 调整形状并归一化
    return img

# 读取并预处理图像
image_path = 'C:/Users/Even/Desktop/Lu/guidebook/minst/image.jpg'
img = preprocess_image(image_path)

# 将 NumPy 数组转换为 TensorFlow 张量
img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

# 进行预测
predict_prob = model(img_tensor)
predicted_class = np.argmax(predict_prob)

print('识别为：', predicted_class)

# 显示图像
image = cv2.imread(image_path)
cv2.imshow("Image1", image)
cv2.waitKey(0)


#批量处理，对文件夹所有图片检测
# import cv2
# import numpy as np
# import os
# import tensorflow as tf

# # 加载 SavedModel
# model = tf.saved_model.load('C:/Users/Even/Desktop/Lu/guidebook/minst/50epochs_train')

# # 定义图像预处理函数
# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(image, (28, 28))  # 调整为模型的输入大小
#     img = img.reshape(-1, 28, 28, 1) / 255.0  # 调整形状并归一化
#     return img

# # 文件夹路径
# folder_path = 'C:/Users/Even/Desktop/Lu/guidebook/minst/test image'

# # 获取文件夹中所有图像文件
# image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# # 对每张图像进行预测
# # 对每张图像进行预测
# for image_file in image_files:
#     # 构造图像完整路径
#     image_path = os.path.join(folder_path, image_file)

#     # 读取并预处理图像
#     img = preprocess_image(image_path)

#     # 将 NumPy 数组转换为 TensorFlow 张量
#     img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

#     # 进行预测
#     predict_prob = model(img_tensor)
#     predicted_class = np.argmax(predict_prob)

#     print(f'{image_file} 识别为：', predicted_class)

#     # 显示图像
#     image = cv2.imread(image_path)
#     cv2.imshow("Image", image)
    
#     # 等待1秒，按任意键继续下一张图片
#     cv2.waitKey(1000)

# # 关闭所有图像窗口
# cv2.destroyAllWindows()


