import imageio
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.python.framework.error_interpolation import interpolate

'''
cv2.imread
    The image is in B,G,R order.
imageio.imread (=deprecated scipy.misc.imread)
    The image is in R,G,B order.

Algorithom
    use np.split and np.concatenate to swift the channel.
'''
def image_channels():
    path='/Users/chendanxia/sophie/inference/WechatIMG46.jpeg'
    img=cv2.imread(path)
    R,G, B=np.split(img,3,axis=2)
    img1=np.concatenate([B,G,R],axis=2)
    img2=imageio.imread(path)
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(img1)
    plt.subplot(133)
    plt.imshow(img2)
    plt.show()

def sort():
    names=['bob','kelly','emma']
    names.sort()
    print(names)

def test():
    dir_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/images/'
    for name in os.listdir(dir_path):
        path=dir_path+name
        img = imageio.imread(path)
        
        plt.subplot(121)
        plt.imshow(img)
        img2=cv2.resize(img,(256,256))
        plt.subplot(122)
        plt.imshow(img2)
        plt.show()
        break;
sort()