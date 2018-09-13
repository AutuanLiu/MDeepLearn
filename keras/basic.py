"""
Email : autuanliu@163.com
Date：2018/9/13
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping

# 获取数据
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 数据预处理
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
# 归一化处理
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# one-hot 编码
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# 构建模型
# model = Sequential([
#     Dense(32, input_shape=(28*28,)),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax')
# ])

# 或者
model = Sequential()
model.add(Dense(32, input_shape=(28*28,))) # 在第一层指定输入的维度，因为后面的层可以自动推断输入维度
# input_shape 和 input_dim 参数是等价的
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# 编译模型
# 包括 优化器 + 损失函数 + 评估标准
# 这些参数都可以是现有方法的字符串标识符也可以是具体类的实现或者函数
model.compile('rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 训练模型
# 在验证集的误差不再下降时, 使用 earlystopping 中断训练
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# 如果您将 model.fit 中的 validation_split 参数设置为 0.1，那么使用的验证数据将是最后 10％ 的数据。如果设置为 0.25，就是最后 25% 的数据。
# 注意，在提取分割验证集之前，数据不会被混洗，因此验证集仅仅是传递的输入中最后一个 x％ 的样本
hist = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
print(hist.history)
model.summary()
keras.utils.plot_model(model, to_file='./keras/model.png')

# 评估模型
score = model.evaluate(x_test, y_test)
# 输出结果
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# 保存模型(HDF5)
model.save('./keras/model01.h5')
model.save_weights('./keras/model01_weights.h5')
model01 = keras.models.load_model('./keras/model01.h5')

# 保存模型的结构
json_str = model.to_json()
yaml_str = model.to_yaml()
print(json_str)
print(yaml_str)

# 冻结网络层
# frozen_layer = Dense(32, trainable=False)
# frozen_layer.trainable = True
# 可以在实例化之后将网络层的 trainable 属性设置为 True 或 False。为了使之生效，在修改 trainable 属性之后，需要在模型上调用 compile()

# 使 RNN 具有状态意味着每批样品的状态将被重新用作下一批样品的初始状态
# 可以通过调用 .pop() 来删除 Sequential 模型中最后添加的层
# 可以使用 keras.applications 模块进行导入预训练的模型
