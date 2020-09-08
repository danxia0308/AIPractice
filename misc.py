import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

print(tf.round(0.4))
print(tf.round(0.51))
print(tf.round(0.6))

def func1():
    a=tf.Variable(initial_value=tf.zeros([2,2]), shape=[2,2], dtype=tf.float32);
    a=tf.Variable(tf.zeros([2,2]));
    a=tf.getVariable('y', shape=[8], initializer=tf.ones_initializer);
    print(a)

def nestTest():
    t = tf.Variable([[1,2],[3,4]])
    print(t)
    print(nest.flatten(t))
    

def _create_zero_arrays(size):
    return array_ops.ones(
        array_ops.stack([2, size]), tf.float32)
    
def tensorArray2():
    time = array_ops.constant(0, dtype=tf.int32, name="time")
    time_steps = 3
    flat_output_size = [4]
    
    output_ta = tuple(tf.TensorArray(dtype=tf.float32, size=time_steps, tensor_array_name="output_%d" % i)
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
    
    time=time+1
    _, output_final_ta = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step_rest,
        loop_vars=(time, output_ta),
        parallel_iterations=32,
        maximum_iterations=100,
        swap_memory=None)
    
    print(output_final_ta)
    print(output_final_ta[0].stack())


def tensorArray():
    ta=tf.TensorArray(dtype=tf.float32, size=2, dynamic_size=True)
    ta=ta.write(0,[0.1,0.2])
    ta=ta.write(1,[1.1,1.2])
    ta=ta.write(2,[2.1,2.2])
    print(ta.stack())
    print('read 0',ta.read(0))
    # print('read 1',ta.read(1))
    # print('read 0',ta.read(0))
    print(ta.stack())
    ta.unstack([0.3,0.4])
    print(ta.stack())
    
    ta = tuple(tf.TensorArray(dtype=tf.float32, size=3, tensor_array_name="output_%d" % i) for i in range(5))
    print(ta)


def while_loop():
    def _time_step2(time, output_ta, state):
        print(time)
        return time+1,output_ta, state
    
    time = array_ops.constant(0, dtype=tf.int32, name="time")
    time_steps = 5
    output_ta = tf.Variable([1,2,3])
    state = tf.Variable([1,2,3])
    parallel_iterations = None
    parallel_iterations = parallel_iterations or 32
    print(parallel_iterations)
    
    _, output_final_ta1, final_state1 = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step2,
        loop_vars=(time, output_ta, state),
        parallel_iterations=parallel_iterations,
        maximum_iterations=100,
        swap_memory=None)
    _, output_final_ta1, final_state1 = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step2,
        loop_vars=(time, output_ta, state),
        parallel_iterations=parallel_iterations,
        maximum_iterations=100,
        swap_memory=None)
    
def const():
    print(tf.constant(1.0, shape=[10]))
    print(tf.constant(1.0))

def eagerly():
    print(tf.executing_eagerly())   # It is eagerly by default.
    a=tf.Variable(3)
    f = tf.cond(tf.greater(a, 0), lambda:True, lambda:False)
    print(f)
        
def lambda_f():
    a=2
    y = lambda x,*_:x>a
    print(y(3))
    foo = [1,2,3,4,5,6,7,8,9,0]
    f = filter(lambda x: x%3==0,foo)
    print(f)
    
    

# tf.enable_eager_execution()
# x=[[2.]]
# m = tf.matmul(x,x)
# print("hello,{}".format(m))


def m2():
    a=[1,3,5]
    b=[2,2,6]
    c=np.where(a>b,a,b)
    print(c)

def m1():
    # a=tf.Variable([1,2,3,6])
    a=[1,2,3,6]
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        max=tf.reduce_max(a)
        print(sess.run(max))
        print(a)
    #     print(sess.run(a))

