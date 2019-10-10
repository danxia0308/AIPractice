'''
dataset = dataset.map(parse_function)    //将dataset通过parse_function进行解析。
dataset = dataset.map(lambda filename, label:tuple(tf.py_func(_read_py_function, [filename,label], [tf.uint8,label.dtype])))    //使用tf.py_func来作用于非tf函数。
训练中使用dataset的顺序：
1. tf.data创建dataset。
2. dataset.map做预处理。
3. dataset.shuffle(buffer_size=1000) 乱序
4. dataset.batch(32)
5. dataset.repeat(num_epochs)
6. dataset.make_one_shot_iterator或者使用make_initializable_iterator并run其initializer。
7. 用iterator的get_next来获取下一个batch，直到出现OutOfRangeError。
'''
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.errors_impl import OutOfRangeError

'''
tf.data.Dataset.from_tensor_slices    输入一个或多个tensor，可为数组或者dict。第一维表示数目，需要保持一致。
tf.data.make_one_shot_iterator        产生一次输出。
dataset.batch                         产生batch数据
'''
def gen_data_and_use():
    dataset=tf.data.Dataset.from_tensor_slices({'a':np.arange(10),'b':np.arange(10,20)});
    dataset=tf.data.Dataset.from_tensor_slices((np.arange(10),np.arange(10,20)));
    dataset=dataset.shuffle(10)
    # 挨个读取
    iter=tf.data.make_one_shot_iterator(dataset)
    e1,e2=iter.get_next()
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run([e1]))
                print(sess.run([e2]))
            except OutOfRangeError as e:
                break
    return
#     batched_dataset = dataset.padded_batch(4, padded_shapes=[None])
    # Batch读取
    batched_dataset = dataset.batch(5)
    iter_batch = batched_dataset.make_one_shot_iterator()
    next_batch=iter_batch.get_next()
    with tf.Session() as sess:
        for i in range(2):
            print(sess.run(next_batch))
    init_iter = dataset.make_initializable_iterator()
    next_element = init_iter.get_next()
    with tf.Session() as sess:
        sess.run(init_iter.initializer)
        while True:
            try:
                print(sess.run(next_element))
            except OutOfRangeError as e:
                break
        sess.run(init_iter.initializer)
        while True:
            try:
                print(sess.run(next_element))
            except OutOfRangeError as e:
                break
    
    
gen_data_and_use()

#TODO What's parse_single_example?
def __parse_function(example_proto):
    features={'image':tf.FixedLenFeature((),tf.string, default_value=""),
              'label':tf.FixedLenFeature((),tf.int64, default_value=0)}
    parsed_features=tf.parse_single_example(example_proto,features)
    return parsed_features['image'],parsed_features['label']

'''
注意dataset在map和batch之后，都需要返回并赋值给新的dataset。
'''
class Agent():
    def __init__(self):
        pass
    def __parse(self, filename, label):
    #     image_string = tf.read_file(filename)
    #     image_decoded = tf.image.decode_jpeg(image_string)
    #     image_resized = tf.image.resize_images(image_decoded,(256,256))
    #     return image_resized, label
        return filename, label
    
    def run(self):
        filenames=['1.jpg','2.jpg']
        dataset=tf.data.Dataset.from_tensor_slices((filenames, filenames)).shuffle(buffer_size=len(filenames))
        dataset = dataset.map(self.__parse,num_parallel_calls=8).batch(2)
        iterator=dataset.make_initializable_iterator()
#         dataset.batch(batch_size=2)
        next_element=iterator.get_next()
        with tf.Session() as sess:
#             iterator=tf.data.make_one_shot_iterator(dataset)
            sess.run(iterator.initializer)
            for i in range(2):
                print(sess.run(next_element))
agent=Agent()
# agent.run()
    