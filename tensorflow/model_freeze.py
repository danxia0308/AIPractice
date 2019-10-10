import tensorflow as tf
import imageio
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import time
from image_process import image_segmentation as seg
import cv2 as cv


checkpoint_dir='/Users/chendanxia/sophie/checkpoint/temp/'
pb_model_file='/Users/chendanxia/sophie/bak/segmentation_model_128.pb'
width=128
height=128

# checkpoint_dir='/Users/chendanxia/sophie/checkpoint/temp/'
# pb_model_file='/Users/chendanxia/sophie/shufflenetv2.pb'
# width=512
# height=512

def convert_checkpoint_to_pb_model():
    state=tf.train.get_checkpoint_state(checkpoint_dir)
    print(state)
    checkpoint=state.model_checkpoint_path
    with tf.Session(graph=tf.Graph()) as sess:
        saver=tf.train.import_meta_graph(checkpoint+'.meta', clear_devices=True)
        saver.restore(sess, checkpoint)
        graph=tf.get_default_graph()
        output_graph_def=tf.graph_util.convert_variables_to_constants(sess,graph.as_graph_def(),output_node_names=['network/output/out_argmax','network/output/out_softmax','network/output/out_probs'])
        
        with tf.gfile.GFile(pb_model_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

def load_graph(pb_model_file):
    with tf.gfile.GFile(pb_model_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='segmentation')
    return graph

def merge_img_and_mask(img, mask):
    a=np.array([mask,mask,mask])
    b=np.transpose(a,[1,2,0])
    c=np.where(b==0,img,255)
    dst_img=np.array(c,np.uint8)
    return dst_img

def merge_img_and_mask_smooth(img, mask):
    a=(1-mask)*255
    dst_img=np.zeros(img.shape,np.uint8)
    color=[255,255,255]
    a=cv.blur(a,(31,31))
    a=a/255.0
    for c in range(3):
        dst_img[:,:,c]=a[:,:]*img[:,:,c]+(1-a[:,:])*color[c]
#         dst_img[:,:,c]=img[:,:,c]
    return dst_img

def get_seg_img(img,probs_out):
#     temp=np.transpose(out_max[0],(2,0,1))
    h, w=img.shape[:2]
#     temp=[temp[0],temp[0],temp[0]]
#     percentage=np.transpose(temp,(1,2,0))
#     percentage=np.where(percentage>0.8,1,percentage)
    temp=np.array([probs_out,probs_out,probs_out])
    percentage=temp.transpose((1,2,0))
    if img.shape[:2] != percentage.shape:
        percentage=cv.resize(percentage, (w,h))
    
    dst_img = img*percentage+255*(1-percentage)
    return dst_img.astype(np.uint8)
    

def load_and_infere():
    graph=load_graph(pb_model_file)
    x_pl=graph.get_tensor_by_name('segmentation/network/input/x_pl:0')
    out_argmax=graph.get_tensor_by_name('segmentation/network/output/out_argmax:0')
    out_softmax=graph.get_tensor_by_name('segmentation/network/output/out_softmax:0')
    out_probs=graph.get_tensor_by_name('segmentation/network/output/out_probs:0')
#     is_training=graph.get_tensor_by_name('segmentation/network/input/Placeholder:0')
    with tf.Session(graph=graph) as sess:
#         img_path='/Users/chendanxia/sophie/sophie.jpg'
        img_path='/Users/chendanxia/sophie/inference/WechatIMG268.jpeg'
        imgs=[]
        img=imageio.imread(img_path)
        resized_img=misc.imresize(img, (width,height))
        imgs.append(resized_img)
        input_imgs=np.array(imgs)
        probs_out, out_arg=sess.run([out_probs, out_argmax], feed_dict={x_pl:input_imgs})
        mask=misc.imresize(out_arg[0], img.shape[:2])
        mask=np.where(mask==255,1,0)
        dst_img=merge_img_and_mask_smooth(img, mask)
        seg_img=get_seg_img(img, probs_out[0])
        plt.subplot(121)
        plt.imshow(dst_img)
        plt.subplot(122)
        plt.imshow(seg_img)
        plt.show()
        

convert_checkpoint_to_pb_model()
# load_and_infere()
