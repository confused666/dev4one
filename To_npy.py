#图片转换成numpy格式
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 加载所有图片
images = []
labels = []
for label in os.listdir('data'):
    if label == '0' or label == '1' or label == '2':
        for file in os.listdir(os.path.join('data', label)):
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                # 加载图片
                img = Image.open(os.path.join('data', label, file))
                # 将图片添加到列表中
                images.append(img)
                labels.append(int(label))

# 将图片转换为 numpy 数组
X = np.array([np.array(img) for img in images])
y = np.array(labels)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3,random_state=42)

# 保存训练集和验证集为npy文件
np.save('data/train_X.npy', train_X)
np.save('data/train_y.npy', train_y)
np.save('data/val_X.npy', val_X)
np.save('data/val_y.npy', val_y)

