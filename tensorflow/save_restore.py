import tensorflow as tf

'''
tf.train.Saver    #创建saver
saver.save        #保存模型
saver.restore     #恢复模型
tf.train.latest_checkpoint    #查找最新的模型文件
使用save保存时，保存了-number.index/-number.meta/-number.data/checkpoint文件。其中meta存储了图结构，data存储了变量，checkpoint保存了最新的模型的number。
'''
def use_saver():
    #max_to_keep表示保留最新的模型个数。0和None表示每个都保存。
    saver = tf.train.Saver(max_to_keep=0)
    
    save_list = tf.trainable_variables()
    global_step=0
    save_list.append(global_step)
    #Saver可以设置需要保存的变量
    saver = tf.train.Saver(save_list,max_to_keep=5)
    
    sess=tf.Session()
    #保存session中的变量到文件
    saver.save(sess,'./my_model.ckpt', global_step=10)
    
    #找到最新的checkpoint并restore。
    save_path=tf.train.latest_checkpoint('checkpoint_dir')
    saver.restore(sess, save_path)
    