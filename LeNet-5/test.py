# import cv2
# import numpy as np
# import tensorflow as tf

# # 加载 SavedModel
# model = tf.saved_model.load('./50epochs_train')

# # 定义图像预处理函数
# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(image, (28, 28))  # 调整为模型的输入大小
#     img = img.reshape(-1, 28, 28, 1) / 255.0  # 调整形状并归一化
#     return img

# # 读取并预处理图像
# image_path = 'LeNet-5/test image/5.jpg'
# # img = preprocess_image(image_path)

# # # 将 NumPy 数组转换为 TensorFlow 张量
# # img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

# # # 进行预测
# # predict_prob = model(img_tensor)
# # predicted_class = np.argmax(predict_prob)

# # print('识别为：', predicted_class)

# # # 显示图像
# # image = cv2.imread(image_path)
# # cv2.imshow("Image1", image)
# # cv2.waitKey(0)


import cv2
import numpy as np
import os
import tensorflow as tf

# 加载 SavedModel
model = tf.saved_model.load('./50epochs_train_NN')


# 定义图像预处理函数
import cv2  
import numpy as np  

#     return new_img
def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 确保图像是白字黑底，如果是黑字白底，需要反转
    if np.mean(image) > 127.5:
        image = 255 - image

    # 二值化处理
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # 计算重心并将重心置中
    moments = cv2.moments(binary_image)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        
    rows, cols = image.shape
    shiftx = cols//2 - cx
    shifty = rows//2 - cy
    
    translation_matrix = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    centered_image = cv2.warpAffine(binary_image, translation_matrix, (cols, rows))

    # 调整图像大小至28x28
    resized_image = cv2.resize(centered_image, (28, 28), interpolation=cv2.INTER_AREA)

    # 归一化处理: 保证背景为0，字符为1
    normalized_image = resized_image / 255.0
    
    # 为模型添加额外的批处理维度和通道维度
    img = normalized_image.reshape(1, 28, 28, 1)
    #img = normalized_image.reshape(1, 28, 28)
    return img
# 文件夹路径
folder_path = 'LeNet-5/test image/'

# 获取文件夹中所有图像文件
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 对每张图像进行预测
# 对每张图像进行预测
for image_file in image_files:
    # 构造图像完整路径
    image_path = os.path.join(folder_path, image_file)

    # 读取并预处理图像
    img = preprocess_image(image_path)

    # 将 NumPy 数组转换为 TensorFlow 张量
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # 进行预测
    predict_prob = model(img_tensor)
    predicted_class = np.argmax(predict_prob)

    print(f'{image_file} 识别为：', predicted_class)

    # 显示图像
    image = cv2.imread(image_path)
    cv2.imshow("Image", image)
    
    # 等待1秒，按任意键继续下一张图片
    cv2.waitKey(1000)

# 关闭所有图像窗口
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import os
# import onnxruntime as ort

# # 加载ONNX模型
# ort_session = ort.InferenceSession('./model.onnx')

# # 定义图像预处理函数
# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(image, (28, 28))  # 调整为模型的输入大小
#     img = img.reshape(1, 1, 28, 28).astype(np.float32) / 255  # 调整形状并归一化
#     # mean = 0.1307
#     # std = 0.3081
#     # # mean = 0.1
#     # # std = 0.5
#     # img = (img - mean ) / (std )  # 这里乘以255是因为原始均值和标准差是基于0-1范围的值
#     return img

# # 文件夹路径
# folder_path = 'LeNet-5/test image/'

# # 获取文件夹中所有图像文件
# image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# # 对每张图像进行预测
# for image_file in image_files:
#     # 构造图像完整路径
#     image_path = os.path.join(folder_path, image_file)

#     # 读取并预处理图像
#     img = preprocess_image(image_path)

#     # 进行预测
#     inputs = {ort_session.get_inputs()[0].name: img}
#     predict_prob = ort_session.run(None, inputs)
#     predicted_class = np.argmax(predict_prob[0])

#     print(f'{image_file} 识别为：', predicted_class)

#     # 显示图像
#     image = cv2.imread(image_path)
#     cv2.imshow("Image", image)
    
#     # 等待1秒，按任意键继续下一张图片
#     cv2.waitKey(1000)

# # 关闭所有图像窗口
# cv2.destroyAllWindows()
