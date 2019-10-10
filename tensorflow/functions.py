'''
operation在用session run之后可以返回numpy数组。
tf.read_file
tf.image.decode_image
tf.image.flip_left_right
tf.one_hot
tf.gather
tf.argmax
tf.train.get_or_create_global_step以及一个训练函数的小例子。
'''
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import math
import numpy as np

'''
训练的小例子，帮助理解global_step和优化过程。
'''
def global_step():
    global_step=tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2)
    a=tf.Variable(3,dtype=tf.float32)
    x=tf.placeholder(dtype=tf.float32,name='x')
    y=tf.placeholder(dtype=tf.float32, name='y')
    loss = tf.abs(tf.multiply(a,tf.square(x))-y)
    train=optimizer.minimize(loss, global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(global_step))
        for i in range(9):
            l,_ = sess.run([loss, train], feed_dict={x:3,y:9})
            print('loss={}'.format(l))
            print(sess.run(global_step))

global_step()

'''
使用tf.argmax, tf.arg_max过期了
'''
def argmax():
    a=np.array([0,1,5,3])
#     m=tf.arg_max(a, dimension=0)
    m=tf.argmax(a,axis=0)
    with tf.Session() as sess:
        print(sess.run(m))

def random():
    r_n=tf.random.normal(shape=(255,255,3))

def random_crop_by_concat():
    path='/Users/chendanxia/sophie/inference/WechatIMG46.jpeg'
    contents=tf.read_file(path)
    img = tf.image.decode_jpeg(contents, channels=3)
    img1 = tf.image.decode_jpeg(contents, channels=1)
    img2 = tf.concat([img,img1], axis=2)
    img2 = tf.image.resize_images(img2,[512,512])
    print(img2.get_shape())
    img2 = tf.image.random_crop(img2, size=[500,500,4])
    img = tf.gather(img2, indices=[0,1,2], axis=2)
    img = tf.cast(img, dtype=tf.uint8)
    img1 = tf.gather(img2, indices=[3], axis=2)
    img1 = tf.cast(img1, dtype=tf.uint8)
    with tf.Session() as sess:
        img_0 = sess.run(img)
        img_1 = sess.run(img1)
        print(img_0)
        plt.imshow(img_0)
        plt.show()

def random_crop_by_expand_dims():
    a=np.zeros((2,2,3))
    b=np.ones((2,2,1))
    a1= tf.expand_dims(a, axis=3)
    b1= tf.expand_dims(b, axis=3)
    c= tf.concat([a1,b1], axis=3)
    a2 = tf.gather(c, axis=3, indices=[0])
    b2 = tf.gather(c, axis=3, indices=[1])
    a2 = tf.squeeze(a2)
     
    with tf.Session() as sess:
        a2,b2= sess.run([a2,b2])
        print(a2)
        print(b2)
        


def flip(img):
    return tf.image.flip_left_right(img)

def onehot():
    num_classes=10
    hot=tf.one_hot([2,3],dtype=tf.float64,depth=num_classes)*[1,2,3,4,5,6,7,8,9,10]
    with tf.Session() as sess:
        hot_run=sess.run(hot)
        print(hot_run)

def gather():
    a=np.ones((2,2,2))
    b=tf.gather(a,axis=2,indices=[0])
    c=tf.squeeze(b)
    print(a.shape)
    print(b)
    print(c)
    with tf.Session() as sess:
        print(sess.run(b))
        print(sess.run(c))

'''
concat:在指定维度上合并元素，相当于对应维度上所有元素内容append在一起。
    必须指定axis。
    指定的axis维度上不能为标量列表。
'''
def concat():
    a=[[[1,2],[1,2],[1,2]],[[3,4],[3,4],[3,4]]]
    b=[[1,2],[3,4]]
    a_c_0=tf.concat(a, axis=0, name='concat1')
    a_c_1=tf.concat(a, axis=1, name='concat1')
    b_c=tf.concat(b, axis=0, name='concat2')
    with tf.Session() as sess:
        print(sess.run(a_c_0))
        print(sess.run(a_c_1))
        print(sess.run(b_c))

def reduceMean():
    a=[1.0,2]
    b=[[2,2],[4,4]]
    a_mean=tf.reduce_mean(a)
    b_mean=tf.reduce_mean(b)
    with tf.Session() as sess:
        print(sess.run(a_mean))
        print(sess.run(b_mean))

def size():
    x_pl = tf.placeholder(tf.uint8,[2, 4, 4, 3], name='x_pl')
    print('shape=',x_pl.get_shape().as_list())
    print('ndims=',x_pl.shape.ndims)
    print(tf.shape(x_pl))
    print(type(x_pl.get_shape()))




'''
用math函数来验证交叉熵损失函数。
TODO：
    实现tf.losses.sparse_softmax_cross_entropy
'''
def cross_entropy():
    num_classes=3
    logits=[]
    logits.append([0.1,0,0.9])
    logits.append([0.2,0,0.8])
    labels=[2,2]
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    with tf.Session() as sess:
        ce_result=sess.run(ce)
        print(ce_result)
    p=math.exp(0.9)/(math.exp(0.1)+math.exp(0.9)+math.exp(0))
    ce_equal=-math.log(p)
    print(ce_equal)
    p=math.exp(0.8)/(math.exp(0.8)+math.exp(0.2)+math.exp(0))
    ce_equal=-math.log(p)
    print(ce_equal)
    p=math.exp(1)/(math.exp(0)+math.exp(0)+math.exp(1))
    ce_equal=-math.log(p)
    print(ce_equal)
    p=math.exp(0)/(math.exp(0)+math.exp(0)+math.exp(1))
    ce_equal=-math.log(p)
    print(ce_equal)

def test():
    dir_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/images/'
    for name in os.listdir(dir_path):
        path=dir_path+name
        img_file=tf.read_file(path)
        img_op = tf.image.decode_image(img_file, 3);
        img_flip_op=flip(img_op)
        with tf.Session() as sess:
            
            img, img_flip=sess.run([img_op, img_flip_op])
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(img_flip)
            print(img_flip_op.get_shape().as_list())
#             plt.show()
            break

# test()