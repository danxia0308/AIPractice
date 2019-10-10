import numpy as np
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def guassian_noise_numpy():
    path='/Users/chendanxia/sophie/inference/WechatIMG46.jpeg'
    img=imageio.imread(path)
    h,w,c=img.shape
    gussian_mean=np.random.random()*10
    gussian_mean=10
    start=time.time()
    for i in range(100):
        noise=np.random.normal(gussian_mean,gussian_mean,(h,w,3))
        img2=img+noise
        img2=np.where(img2> 255,255,img2)
        img2=img2.astype(np.uint8)
    end=time.time()
    print(end-start)
    
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()

def guassian_noise_tf():
    path='/Users/chendanxia/sophie/inference/WechatIMG46.jpeg'
    img=imageio.imread(path)
#     img_placeholder=tf.placeholder(dtype=tf.float32, shape=img.shape)
    img1=img.astype(np.float32)
    scale=tf.placeholder(dtype=tf.float32)
    img_with_noise=tf.add(img1, scale*tf.random_normal(img.shape))
    img_with_noise2=tf.cast(img_with_noise,tf.uint8)
    with tf.Session() as sess:
        start=time.time()
        for i in range(100):
            img2=sess.run(img_with_noise2, feed_dict={scale:np.random.random()*10})
        end=time.time()
        print(end-start)
        
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()

guassian_noise_tf()
# guassian_noise_numpy()