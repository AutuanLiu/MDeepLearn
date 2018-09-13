"""
Email : autuanliu@163.com
Date：2018/9/13
"""

# keras 的函数式 API
from keras.layers import Input, Dense
from keras.models import Model

# 层的实例是可以调用的
inputs = Input(shape=(784,))
print(inputs)

layer1 = Dense(64, activation='relu')
layer2 = Dense(10, activation='softmax') # 返回值是 实例
out1 = layer1(inputs) # 返回值是 tensor
out2 = layer2(out1)
print(layer1, '\n', out1)
print(layer2, '\n', out2)

model = Model(inputs=inputs, outputs=out2)
model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model)

# 利用函数式 API，可以轻易地重用训练好的模型：可以将任何模型看作是一个层，然后通过传递一个张量来调用它。
# 注意，在调用模型时，您不仅重用模型的结构，还重用了它的权重

# Embedding 层将输入序列编码为一个稠密向量的序列
# LSTM 层把向量序列转换成单个向量，它包含整个序列的上下文信息
# 要在不同的输入上共享同一个层，只需实例化该层一次，然后根据需要传入你想要的输入即可
# 每当你在某个输入上调用一个层时，都将创建一个新的张量（层的输出），并且为该层添加一个节点，将输入张量连接到输出张量

