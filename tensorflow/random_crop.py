import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt; 
import random
import os
import imageio

def crop_method1(img):
    height, width, channel = img.shape
    height_range = height//10
    width_range=width//10
    left_margin=random.randint(0,width_range)
    width_cropped=random.randint(width-height_range*2,width-left_margin)
    top_margin=random.randint(0,height_range)
    height_cropped=random.randint(height-height_range*2,height-top_margin)
    img_cropped=img[top_margin:height_cropped, left_margin:width_cropped]
    return img_cropped

def crop_tf(image):
    height, width, _ = image.shape
    height_range = height//10
    width_range=width//10
    left_margin=random.randint(0,width_range)
    width_cropped=random.randint(width-height_range*2,width-left_margin)
    top_margin=random.randint(0,height_range)
    height_cropped=random.randint(height-height_range*2,height-top_margin)
    croped_img = tf.image.crop_to_bounding_box(image, top_margin, left_margin, height_cropped, width_cropped)
    with tf.Session() as sess:
        img2 =  sess.run(croped_img)
        print(image.shape,img2.shape)
        return img2

def crop_img():
    dir_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/images/'
    for name in os.listdir(dir_path):
        path=dir_path+name
        img = imageio.imread(path)
        
        plt.subplot(221)
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(crop_method1(img))
        plt.subplot(223)
        plt.imshow(crop_tf(img))
        plt.subplot(224)
        plt.imshow(crop_tf(img))
        plt.show()
        break

crop_img()
        