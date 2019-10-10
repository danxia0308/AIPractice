import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
from scipy import misc

'''
无论是用imageio还是cv2读取，其shape顺序为H,W,C。
'''
def width_height():
    img_path='/Users/chendanxia/sophie/test/car.png'
    img=imageio.imread(img_path)
    height,width,channel=img.shape
    print('height={} width={} channel={}'.format(height,width,channel))
    img1=cv2.imread(img_path)
    height,width,channel=img1.shape
    print('height={} width={} channel={}'.format(height,width,channel))
    img2=img[:,0:width//2]
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()
    imageio.imsave('./car.png',img2)
'''
scipy.misc.imresize，输入shape为(height, width)
cv2.resize，输入shape为(width,height)
'''
def resize():
    img_path='/Users/chendanxia/sophie/inference/r.jpg'
    img=imageio.imread(img_path)
    print('img shape={}'.format(img.shape))
    small_image=misc.imresize(img, (256,256))
    img1=misc.imresize(small_image, img.shape[:2])
    img2=cv2.resize(small_image,img.shape[:2])
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(small_image)
    plt.subplot(223)
    plt.imshow(img1)
    plt.subplot(224)
    plt.imshow(img2)
    plt.show()

resize()