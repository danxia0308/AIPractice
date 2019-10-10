import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import misc

# img_path='/Users/chendanxia/sophie/xiaoyu_imgs_segment/9970_87745c42-ecab-4ef8-a705-1f0dbbb555b0.png'
# img=cv.imread(img_path, -1)
# print(img.shape)
img0=cv.imread('/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/masks/ADE_train_00000961_seg.png')
img0=np.where(img0==1,255,0)
# print(img0)
img=cv.imread('/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/masks_by_faceplus/ADE_train_00000961.jpg_seg.png')
img1=np.where(img>255/2,255,0)
img2=np.where(img>100,255,0)
img3=np.where(img>10,255,0)
# COCO_train2014_000000291098.jpg_seg.png
# img3=cv.resize(img, (256,256))
# img4=cv.resize(img2, img.shape)
# plt.subplot(221)
# plt.imshow(img)
# plt.subplot(222)
# plt.imshow(img1)
# plt.subplot(223)
# plt.imshow(img2)
# plt.subplot(224)
# plt.imshow(img0)
# img=cv.imread('/Users/chendanxia/sophie/2.jpg')
# img=cv.resize(img,(255,255))
# cv.imwrite('/Users/chendanxia/sophie/3.jpg',img)
img1=imageio.imread('/Users/chendanxia/sophie/4.png')

print(img1.shape)
img2 = misc.imresize(img1, (577, 433))
plt.imshow(img2)
plt.show()

# cv.imshow('Bye Saw Tooth',img)
# cv.waitKey()