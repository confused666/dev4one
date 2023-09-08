import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import os
import tensorflow as tf
import time

model = tf.keras.models.load_model('model/transfer_resnet_3class.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

def toarray(image):
    # 调整图像大小
    image = image.resize((512, 512))
    # 创建一个512x512的黑色背景
    background = Image.new('RGB', (512, 512), (0, 0, 0))
    # 将图像嵌入到黑色背景中心
    x_offset = (512 - image.width) // 2
    y_offset = (512 - image.height) // 2
    background.paste(image, (x_offset, y_offset))
    #将图像转换为NumPy数组
    array = np.asarray(image)
    # 将数组数据类型转换为float32
    array = array.astype('float32')
    return array
def forward(array):
    # 加载模型
    # 使用模型进行预测等操作
    # 调整数组形状以匹配模型输入形状
    array = array.reshape((1, array.shape[0], array.shape[1], 3))
    # 将数组归一化到0到1之间
    # array /= 255.0
    # 使用模型进行预测
    predictions = model.predict(array)
    predictions = np.argmax(predictions)
    return predictions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.files['fileInput'].filename != '':
        start_time = time.time()
        file = request.files['fileInput']
        filename = file.filename
        pre_result = None
        pre_time = None
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        array = toarray(img)
        pre_result = forward(array)
        end_time = time.time()
        pre_time = end_time - start_time
    else:
        start_time = time.time()
        file = request.files['cameraInput']
        filename = file.filename
        pre_result = None
        pre_time = None
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        array = toarray(img)
        pre_result = forward(array)
        end_time = time.time()
        pre_time = end_time - start_time

    if pre_result == 0:
        pre_result = '合格'
    elif pre_result == 1:
        pre_result = '不合格'
    elif pre_result == 2:
        pre_result = '图片不完整,无法判断'

    return render_template('upload.html', filename=filename, pre_result=pre_result, predict_time=pre_time)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
