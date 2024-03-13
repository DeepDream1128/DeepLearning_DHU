import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical

def get_local_mnist_data():
    # 读取本地的MNIST数据集文件
    train_images_file = './Data/train-images.idx3-ubyte'
    train_labels_file = './Data/train-labels.idx1-ubyte'
    test_images_file = './Data/t10k-images.idx3-ubyte'
    test_labels_file = './Data/t10k-labels.idx1-ubyte'
    # 使用idx2numpy读取数据集文件
    x_train_original = idx2numpy.convert_from_file(train_images_file)
    y_train_original = idx2numpy.convert_from_file(train_labels_file)
    x_test_original = idx2numpy.convert_from_file(test_images_file)
    y_test_original = idx2numpy.convert_from_file(test_labels_file)

    # 从训练集中分配验证集
    x_val = x_train_original[55000:]
    y_val = y_train_original[55000:]
    x_train = x_train_original[:55000]
    y_train = y_train_original[:55000]

    # 将图像转换为四维矩阵(nums,rows,cols,channels)，这里把数据从uint8类型转化为float32类型，提高训练精度。
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test_original.reshape(x_test_original.shape[0], 28, 28, 1).astype('float32')

    # 原始图像的像素灰度值为0-255，为了提高模型的训练精度，通常将数值归一化映射到0-1。
    x_train = x_train / 255
    x_val = x_val / 255
    x_test = x_test / 255

    # 图像标签一共有10个类别即0-9，这里将其转化为独热编码（One-hot）向量
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test_original, 10)

    return x_train, y_train, x_val, y_val, x_test, y_test

# 调用函数获取本地数据集
x_train, y_train, x_val, y_val, x_test, y_test = get_local_mnist_data()

"""
定义LeNet-5网络模型
"""
def LeNet5():

    input_shape = Input(shape=(28, 28, 1))

    x = Conv2D(6, (5, 5), activation="relu", padding="same")(input_shape)
    x = MaxPooling2D((2, 2), 2)(x)
    x = Conv2D(16, (5, 5), activation="relu", padding='same')(x)
    x = MaxPooling2D((2, 2), 2)(x)

    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(input_shape, x)
    print(model.summary())

    return model

"""
编译网络并训练
"""
x_train, y_train, x_val, y_val, x_test, y_test = get_local_mnist_data()
model = LeNet5()

# 编译网络（定义损失函数、优化器、评估指标）
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 开始网络训练（定义训练数据与验证数据、定义训练代数，定义训练批大小）
train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32, verbose=2)

# 模型保存
model.save('50epochs_train')

# 定义训练过程可视化函数（训练集损失、验证集损失、训练集精度、验证集精度）
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

# 输出网络在测试集上的损失与精度
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 测试集结果预测
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
print('前20张图片预测结果：', predictions[:20])

# 预测结果图像可视化
_,_,_,_,x_test_original,y_test_original = get_local_mnist_data()
def mnist_visualize_multiple_predict(start, end, length, width):

    for i in range(start, end):
        plt.subplot(length, width, 1 + i)
        plt.imshow(x_test_original[i], cmap=plt.get_cmap('gray'))
        title_true = 'true=' + str(y_test_original[i])
        # title_prediction = ',' + 'prediction' + str(model.predict_classes(np.expand_dims(x_test[i], axis=0)))
        title_prediction = ',' + 'prediction' + str(predictions[i])
        title = title_true + title_prediction
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show()

mnist_visualize_multiple_predict(start=0, end=9, length=3, width=3)
if y_test_original.ndim > 1:  # 独热编码数组维度大于1
    y_test_original = np.argmax(y_test_original, axis=1)

# 如果predictions是独热编码，同样转换为整数标签
if predictions.ndim > 1:
    predictions = np.argmax(predictions, axis=1)
# 混淆矩阵
cm = confusion_matrix(y_test_original, predictions)
cm = pd.DataFrame(cm)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, cmap='Oranges', linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(cm)