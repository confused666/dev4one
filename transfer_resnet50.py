#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import Input, Flatten, Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# In[2]:
# 加载数据
train_X = np.load('data_original/train_X_3.npy')
print(train_X.shape)
train_y = np.load('data_original/train_y_3.npy')
train_y = tf.keras.utils.to_categorical(train_y)

val_X = np.load('data_original/val_X_3.npy')
print(val_X.shape)
val_y = np.load('data_original/val_y_3.npy')
val_y = tf.keras.utils.to_categorical(val_y)

# In[3]:
#使用预训练权重
inputs = Input(shape=(512, 512, 3))
resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs, input_shape=(512, 512, 3))
#冻结所有层
for layer in resnet.layers:
    layer.trainable = False
x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(3, activation='softmax')(x)
model = Model(inputs=resnet.input, outputs=x)

# In[4]:
# 创建ModelCheckpoint回调函数
checkpoint = ModelCheckpoint('resnet_model_3class.h5', monitor='val_accuracy', save_best_only=True, mode='max')
# 创建Adam优化器对象，并指定初始学习率为0.001
optimizer = Adam(learning_rate=0.001)
# 编译模型
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
history = model.fit(train_X, train_y, batch_size=32, epochs=20, validation_data=(val_X, val_y), callbacks=[checkpoint])

# In[5]:
# 保存训练过程中的损失和准确率数据
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
# 将数据保存到文件中
np.savetxt("full_train_loss.txt", train_loss, delimiter=",")
np.savetxt("full_train_acc.txt", train_acc, delimiter=",")
np.savetxt("full_val_loss.txt", val_loss, delimiter=",")
np.savetxt("full_val_acc.txt", val_acc, delimiter=",")               

# In[6]:
# 绘制模型训练曲线和验证曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('full_resnet_acc.jpg')
plt.show()

# In[7]:
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('full_resnet_loss.jpg')
plt.show()

# In[8]:
#解冻40层之后的进行训练
model = tf.keras.models.load_model('resnet_model_3class.h5')
for layer in resnet.layers[:40]:
    layer.trainable = False
for layer in resnet.layers[40:]:
    layer.trainable = True

checkpoint = ModelCheckpoint('transfer_resnet_model_3class.h5', monitor='val_accuracy', save_best_only=True, mode='max')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=32, epochs=20, callbacks=[checkpoint])

# In[9]:
# 保存训练过程中的损失和准确率数据
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
# 将数据保存到文件中
np.savetxt("train_loss.txt", train_loss, delimiter=",")
np.savetxt("train_acc.txt", train_acc, delimiter=",")
np.savetxt("val_loss.txt", val_loss, delimiter=",")
np.savetxt("val_acc.txt", val_acc, delimiter=",")               

# In[10]:
# 绘制模型训练曲线和验证曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('resnet_acc.jpg')
plt.show()

# In[11]:
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('resnet_loss.jpg')
plt.show()










