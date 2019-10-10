from skimage import exposure
import numpy as np
import imageio
import matplotlib.pyplot as plt

'''
Use skimage.exposure.adjust_gamma to adjust contrast.
    if gamma > 1, then contrast is lower. 
    if gamma < 1, then contrast is higher
Use skimage.exposure.is_low_contrast to check if the contrast is too low.
'''
def gamma_adjust():
    path='/Users/chendanxia/sophie/inference/WechatIMG46.jpeg'
    img=imageio.imread(path)
    img1=exposure.adjust_gamma(img,0.5)
    img2=exposure.adjust_gamma(img,1.5)
    img3=exposure.adjust_gamma(img,2)
    print(exposure.is_low_contrast(img))
    print(exposure.is_low_contrast(img1))
    print(exposure.is_low_contrast(img2))
    print(exposure.is_low_contrast(img3))
    
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(img1)
    plt.subplot(223)
    plt.imshow(img2)
    plt.subplot(224)
    plt.imshow(img3)
    plt.show()
#     imageio.imsave('/Users/chendanxia/sophie/inference/1.jpg',img)
#     imageio.imsave('/Users/chendanxia/sophie/inference/2.jpg',img1)
#     imageio.imsave('/Users/chendanxia/sophie/inference/3.jpg',img2)
#     imageio.imsave('/Users/chendanxia/sophie/inference/4.jpg',img3)

gamma_adjust()
    