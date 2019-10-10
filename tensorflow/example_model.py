import tensorflow as tf
import imageio
import matplotlib.pyplot as plt 
import numpy as np
from scipy import misc
from tensorflow.python.util.compat import as_text
from bleach._vendor.html5lib._ihatexml import name

pb_model_path='/Users/chendanxia/sophie/checkpoint/testmodel.pb'
def build_save_model():
    with tf.Session() as sess:
        input_x=tf.placeholder(tf.int16,[256,256,3],name='input_x')
        output_x=tf.cast(input_x, dtype=tf.uint8,name='output_x')
        graph=tf.get_default_graph()
        output_graph_def=tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(),output_node_names=['output_x'])
        with tf.gfile.GFile(pb_model_path,"wb") as f:
            f.write(output_graph_def.SerializeToString())

def load_graph(pb_model_file):
    with tf.gfile.GFile(pb_model_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='example')
    return graph

def inference():
    graph=load_graph(pb_model_path)
    input_x=graph.get_tensor_by_name('example/input_x:0')
    output_x=graph.get_tensor_by_name('example/output_x:0')
    with tf.Session(graph=graph) as sess:
        img_path='/Users/chendanxia/sophie/inference/hui.jpg'
        img=imageio.imread(img_path)
        img=misc.imresize(img,(256,256))
        imgs=[]
        imgs.append(img)
        out=sess.run(output_x, feed_dict={input_x:np.array(img,np.uint8)})
    img_out=out
    plt.imshow(img_out)
    plt.show()

def build_save_model2():
    with tf.Session() as sess:
        a=tf.Variable(5.0, name='a')
        b=tf.Variable(6.0, name='b')
        c=tf.multiply(a, b, name='c')
        
        a1 = tf.placeholder(tf.float32, name='a1')
        b1 = tf.placeholder(tf.float32, name='b1')
        c1=tf.multiply(a1,b1,name='c1')
        
        a2 = tf.placeholder(tf.uint8, shape=[2], name='a2')
        b2 = tf.placeholder(tf.uint8, shape=[2], name='b2')
        c2 = tf.add(a2, b2, name='c2')
        
        sess.run(tf.global_variables_initializer())
        
        tf.train.write_graph(sess.graph_def, '/Users/chendanxia/sophie/checkpoint', 'testmodel2.pb', as_text=False)

build_save_model()
# inference()
# build_save_model2()
# inference()        
