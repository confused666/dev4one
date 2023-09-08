#数据增强
import os
import numpy as np
from PIL import Image, ImageEnhance

#数据增强生成新图片
def brightnessEnhancement(root_path,img_name):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.1+0.2*np.random.random()#取值范围1.1-1.3
    # brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.1+0.2*np.random.random()#取值范围1.1-1.3
    # contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    random_angle = np.random.randint(-2, 2)*90
    if random_angle==0:
     rotation_img = img.rotate(-90) #旋转角度
    else:
        rotation_img = img.rotate( random_angle)  # 旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def createImage(imageDir,saveDir):
    i = 0
    for name in os.listdir(imageDir):
        if name.split('.')[-1] == 'jpg' or name.split('.')[-1] == 'jpeg':
            i = i + 1
            saveName = "cesun" + str(i) + ".jpg"
            saveImage = contrastEnhancement(imageDir, name)
            saveImage.save(os.path.join(saveDir, saveName))
            saveName1 = "flip" + str(i) + ".jpg"
            saveImage1 = flip(imageDir, name)
            saveImage1.save(os.path.join(saveDir, saveName1))
            saveName2 = "brightnessE" + str(i) + ".jpg"
            saveImage2 = brightnessEnhancement(imageDir, name)
            saveImage2.save(os.path.join(saveDir, saveName2))
            saveName3 = "rotate" + str(i) + ".jpg"
            saveImage = rotation(imageDir, name)
            saveImage.save(os.path.join(saveDir, saveName3))

imageDir = "data/0"
saveDir = "data/0"
createImage(imageDir,saveDir)


