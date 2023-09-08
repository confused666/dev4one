#对原始数据进行分类
import os
import shutil
import pandas as pd

df = pd.read_excel('classes.xlsx')
classes = df['class'].tolist()

# 创建类别对应的文件夹
for cls in classes:
    os.makedirs(str(cls), exist_ok=True)

for i in range(129):
    img_file = f'{i:03d}.jpg'
    if not os.path.exists(img_file):
        continue  # 如果文件不存在，则跳过该文件
    else:
        cls_idx = int(img_file.split('.')[0])  # 提取文件名中的数字部分，并转换为整数
        cls_name = classes[cls_idx]  # 获取该图片的类别名称
        dst_folder = f'{cls_name}'
        shutil.copy(img_file, dst_folder)