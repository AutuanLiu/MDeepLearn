"""
Email : autuanliu@163.com
Date：2018/9/13
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D
from keras.optimizers import SGD

# 生成虚拟数据
x_train = np.random.random((500, 100, 100, 3))
x_test = np.random.random((50, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(500, 1)), num_classes=10)
y_test = keras.utils.to_categorical(np.random.randint(10, size=(50, 1)), num_classes=10)

# 构建模型
model = Sequential()
# 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
# 使用 32 个大小为 3x3 的卷积滤波器
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 编译模型
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=150, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print(f'Test loss: {score[0]}\nTest accuracy: {score[1]}')
