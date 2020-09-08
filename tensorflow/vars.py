import tensorflow as tf
import argparse
from tensorflow import pywrap_tensorflow
import os
import numpy as np
from shutil import copyfile


# parser
parser = argparse.ArgumentParser(description='Convert Precision Model')
parser.add_argument('--ckpt', type=str, default='/mnt/data/danxiachen/dnn_save_path/ckpt_noshuffDIEN3',
                    help='path to convert checkpoint file directory (default:'')')
args = parser.parse_args()

def test():
#     checkpoint_dir='/mnt/data/danxiachen/dnn_save_path/'
#     state=tf.train.get_checkpoint_state(checkpoint_dir)
#     with tf.Session() as sess:
#         saver=tf.train.import_meta_graph(state.model_checkpoint_path+'.meta', clear_devices=True)
    reader=tf.train.NewCheckpointReader(args.ckpt)
    print(reader.debug_string().decode("utf-8"))

def print_vars(data_type=np.float16):
    checkpoint_name='ckpt_noshuffDIEN3'
    curent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/'
    out_dir = curent_dir + "model-F16"+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
        
    reader=pywrap_tensorflow.NewCheckpointReader(args.ckpt)
    var_to_map = reader.get_variable_to_shape_map()
    val_f = {}
    for key, dim in var_to_map.items():
        val_f[key.strip(":0")] = tf.Variable(reader.get_tensor(key).astype(data_type))
     #get parameters before convert
    param_log_origin=''
    for key in var_to_map:
        param_log_origin += "tensor_name: "+key+"  shape:"+str(reader.get_tensor(key).shape)+"\r\n"
        param_log_origin += str(reader.get_tensor(key))+"\r\n"  
    writer = open(out_dir+'Param-'+str(reader.get_tensor(key).dtype)+'.txt', 'w', encoding="utf-8")
    writer.write(param_log_origin)      
  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph(args.ckpt+'.meta')
        new_saver.restore(sess,args.ckpt)  
        saver = tf.train.Saver(val_f)
        saver.save(sess, out_dir+checkpoint_name)  

    #save parameters after convert
    reader_convert = pywrap_tensorflow.NewCheckpointReader(out_dir+checkpoint_name)
    var_to_map_convert = reader_convert.get_variable_to_shape_map()  
    param_log_convert=''
    for item in var_to_map_convert:
        param_log_convert += "tensor_name: "+item+"  shape:"+str(reader_convert.get_tensor(item).shape)+"\r\n"
        param_log_convert += str(reader_convert.get_tensor(item))+"\r\n" 
    writer = open(out_dir+'Param-'+str(reader_convert.get_tensor(item).dtype)+'.txt', 'w', encoding="utf-8")
    writer.write(param_log_convert)      
    
    print("Convert Finish!")
    print("Save to path:"+out_dir) 

print_vars()