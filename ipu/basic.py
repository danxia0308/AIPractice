from tensorflow.python.ipu import ipu_optimizer
import tensorflow as tf
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

import numpy as np

TRAIN_DATA_SIZE=1024
BS=1
SEED=3

def data_generator():
    for i in range(TRAIN_DATA_SIZE):
        yield (i % 10,i)

def get_data_set():
    dataset = tf.data.Dataset.from_generator(lambda:data_generator(), (tf.int32,tf.int32), (tf.TensorShape([]),tf.TensorShape([])))
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(BS, drop_remainder=True)
    dataset = dataset.prefetch(10)
    return dataset

def method1(x):
    return tf.add(x,1)

def method2(x):
    return tf.add(x,2)

def method3(i, x):
    i=i+1
    return i, tf.add(x,3)

def model1(x, y,time_steps_ph):
    i = 0
    z1 = control_flow_ops.cond(2 > x[0], lambda:method1(x[0]), lambda:method2(x[0]))
    z2 = control_flow_ops.cond(0 > y[0], lambda:method1(y[0]), lambda:method2(y[0]))
    max_y = math_ops.reduce_max(y)
    i, z2 = control_flow_ops.while_loop(cond=lambda i, *_:i < time_steps_ph, 
        body=method3, 
        loop_vars=(i, z2))
    return z1, z2

def _create_zero_arrays(size):
    return array_ops.ones(
        array_ops.stack([2, size]), tf.float32)
    
def model2(time_steps_ph):
    time = array_ops.constant(0, dtype=tf.int32, name="time")
    flat_output_size = [4]
    
    output_ta = tuple(tf.TensorArray(dtype=tf.float32, size=time_steps_ph, tensor_array_name="output_%d" % i)
                    for i in range(len(flat_output_size)))
    
    flat_zero_output = tuple(_create_zero_arrays(output) for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=36, flat_sequence=flat_zero_output)
    zero_output_ta = tuple(zero_output for i in range(len(output_ta)))
    
    def _time_step_rest(time, output_ta_t):
        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, zero_output_ta))
        return (time+1, output_ta_t)
    
    
    _, output_final_ta = control_flow_ops.while_loop(
        cond=lambda time, *_: time < 1,
        body=_time_step_rest,
        loop_vars=(time, output_ta),
        parallel_iterations=32,
        maximum_iterations=100,
        swap_memory=None)
    
#     time=time+1
    _, output_final_ta = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps_ph,
        body=_time_step_rest,
        loop_vars=(time, output_ta),
        parallel_iterations=32,
        maximum_iterations=100,
        swap_memory=None)
    
    print(output_final_ta)
    print(output_final_ta[0].stack())
    return output_final_ta[0].stack()

def train():
    graph = tf.Graph()
    with graph.as_default():
        dataset = tf.data.Dataset.from_tensors(tf.constant(1, shape=[]))
#         dataset = tf.data.Dataset.from_tensors(np.array([1,2,3,4,5,6,7,8,9,0]))
        dataset = dataset.map(lambda x: [x,x])
        dataset = dataset.batch(BS, drop_remainder=True)
        dataset = dataset.repeat()
        infeed_queue = ipu_infeed_queue.IPUInfeedQueue(get_data_set(), feed_name="infeed")
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name='outfeed')
        time_steps_ph = tf.placeholder(tf.int32, shape=[])
        with ipu_scope('/device:IPU:0'):
            def compile_fn():

                def body(x,y):
#                     z1, z2 = model1(x, y, time_steps_ph)
#                     outfeed = outfeed_queue.enqueue({'z1':z1, 'z2':z2})
                    z3 = model2(time_steps_ph)
                    outfeed = outfeed_queue.enqueue({'z3':z3})
                    return outfeed
                return loops.repeat(1, body,[],infeed_queue)
    
        utils.move_variable_initialization_to_cpu()
        init = tf.global_variables_initializer()
        outputs = ipu_compiler.compile(compile_fn,[])
    
        dequeue_outfeed = outfeed_queue.dequeue()
    ipu_options = utils.create_ipu_config(profiling=False,
                                          profile_execution=False,
                                          max_cross_replica_sum_buffer_size=10000000,
                                          max_inter_ipu_copies_buffer_size=10000000)
    ipu_options = utils.auto_select_ipus(ipu_options,1)
    utils.configure_ipu_system(ipu_options)
    utils.reset_ipu_seed(SEED)
    
    sess = tf.Session(graph=graph)
    sess.run(init)
    sess.run(infeed_queue.initializer)
    
    steps = 6;
    i=0;
    while i < steps:
        sess.run(outputs, feed_dict={time_steps_ph:3})
        result = sess.run(dequeue_outfeed)
        print(result)
        i = i+1
        break

train()