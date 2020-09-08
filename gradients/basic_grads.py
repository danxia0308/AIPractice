# coding=UTF-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from _operator import concat


def gru_unit():
    with tf.compat.v1.Session() as sess:
        inputs = tf.Variable([[1., 1.], [1., 1.], [1., 1.]], dtype=tf.float32, name='inputs')
        state = tf.Variable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=tf.float32, name='state')
        _gate_kernel = tf.Variable([[-0.47945526, 0.29105264, 0.36324948, 0.34305102],
                                    [ 0.3866263, -0.55099547, -0.20881158, 0.79369456],
                                    [-0.78809285, 0.47519642, 0.48400682, 0.16632384],
                                    [-0.7381171, 0.77089626, -0.57933414, -0.29082513]], dtype=tf.float32, name='gate_kernel')
        _gate_bias = tf.Variable([1., 1., 1., 1.], dtype=tf.float32, name='gate_bias')
        _candidate_kernel = tf.Variable([[-0.55362725, 0.33607864],
                                         [ 0.41944432, 0.39612126],
                                         [ 0.4464376, -0.63623476],
                                         [-0.24111485, 0.9164796 ]], dtype=tf.float32, name='cadidate_kernel')
        _candidate_bias = tf.Variable([0.0, 0.0], dtype=tf.float32, name='candidate_bias')
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), _gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, _gate_bias)
        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state
        candidate0 = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), _candidate_kernel)
        candidate = nn_ops.bias_add(candidate0, _candidate_bias)
        c = math_ops.tanh(candidate)
        new_h = u * state + (1 - u) * c
        sess.run(tf.global_variables_initializer())
        print(sess.run(new_h))
        print('candidate', sess.run(candidate))
        print('c', sess.run(c))
        print('u', sess.run(u))
        print('r', sess.run(r))
        loss = tf.reduce_mean(new_h - 1)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = optimizer.compute_gradients(loss)
#         train_op = optimizer.apply_gradients(grads_and_vars)
#         for gv in grads_and_vars:
#             print(gv, sess.run(gv))
        print('loss/new_h', sess.run(tf.gradients(loss, new_h)))
        print('loss/c', sess.run(tf.gradients(loss,c)))
        print('loss/u', sess.run(tf.gradients(loss,u)))
        g_candidate = sess.run((1 - u)*(1-c*c))*sess.run(tf.gradients(loss, new_h))
        print('loss/new_h*(1-u)*(1-c*c)', np.squeeze(g_candidate))
        print('loss/candidate', sess.run(tf.gradients(loss, candidate)))
        print('loss/candidate0', sess.run(tf.gradients(loss, candidate0)))
        print('loss/candidate_bias', sess.run(tf.gradients(loss, _candidate_bias)))
        #g_candiate_kernel
        concat_input = sess.run(array_ops.concat([inputs, r_state], 1))
        print('array_ops.concat([inputs, r_state], 1)', concat_input)
        g_c_kernel = math_ops.matmul(np.transpose(concat_input) , g_candidate)
        print('g_c_kernel', sess.run(g_c_kernel))
        print('loss/_candidate_kernel', sess.run(tf.gradients(loss, _candidate_kernel)))
        #g_r_state
        print('loss/r_state', sess.run(tf.gradients(loss, r_state)))
        g_input_rstate = math_ops.matmul( np.squeeze(g_candidate),np.transpose(sess.run(_candidate_kernel)))
        g_input_candidate, g_r_state = array_ops.split(value=g_input_rstate, num_or_size_splits=2, axis=1)
        print('g_r_state', sess.run(g_r_state))
        
        #g_r
        print('loss/r', sess.run(tf.gradients(loss, r)))
        g_r = sess.run(g_r_state)*sess.run(state)
        print('g_r', g_r)
        #g_u
        g_u = sess.run(state-c)*sess.run(tf.gradients(loss, new_h))
        print('loss/u', sess.run(tf.gradients(loss, u)))
        print('g_u', g_u)
        #g_gate_inputs
        print('loss/gate_inputs', sess.run(tf.gradients(loss, gate_inputs)))
        g_gate_inputs = array_ops.concat([g_r, np.squeeze(g_u)], 1) * value * (1-value)
        print('g_gate_inputs', sess.run(g_gate_inputs))
        g_gate_inputs_state = math_ops.matmul(g_gate_inputs, np.transpose(sess.run(_gate_kernel)))
        g_inputs_gate, g_state_gate = array_ops.split(value=g_gate_inputs_state, num_or_size_splits=2, axis=1)
        print('g_inputs_gate', g_inputs_gate)
        #g_gate_kernel
        print('loss/gate_kernel', sess.run(tf.gradients(loss, _gate_kernel)))
        g_gate_kernel = math_ops.matmul(np.transpose(concat_input), g_gate_inputs)
        print('g_gate_kernel', sess.run(g_gate_kernel))
        #g_gate_bias
        print('loss/gate_bias', sess.run(tf.gradients(loss, _gate_bias)))
        g_gate_bias = tf.reduce_sum(g_gate_inputs, axis=0)
        print('g_gate_bias', sess.run(g_gate_bias))
        print('loss/inputs', sess.run(tf.gradients(loss, inputs)))
        g_input=g_inputs_gate + g_input_candidate
        print('g_inputs', sess.run(g_input))

def augru_unit():
    with tf.compat.v1.Session() as sess:
        inputs = tf.Variable([[1., 1.], [1., 1.], [1., 1.]], dtype=tf.float32, name='inputs')
        state = tf.Variable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=tf.float32, name='state')
        att_score = tf.Variable([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=tf.float32, name='state')
        _gate_kernel = tf.Variable([[-0.12287354,  0.8648142,   0.3188401,   0.8256132 ],
                                    [-0.75681853,  0.4331085,  -0.29249465,  0.65579015],
                                    [-0.08258069, -0.21897888, -0.20116574,  0.52735907],
                                    [ 0.28278595,  0.13071388, -0.43361932, -0.62175727]], dtype=tf.float32, name='gate_kernel')
        _gate_bias = tf.Variable([1., 1., 1., 1.], dtype=tf.float32, name='gate_bias')
        _candidate_kernel = tf.Variable([[-0.17983055, -0.45415568],
                                         [-0.07700777, -0.47652483],
                                         [-0.54514384,  0.32709408],
                                         [-0.29585528, -0.14058399]], dtype=tf.float32, name='cadidate_kernel')
        _candidate_bias = tf.Variable([0.0, 0.0], dtype=tf.float32, name='candidate_bias')
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), _gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, _gate_bias)
        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state
        candidate0 = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), _candidate_kernel)
        candidate = nn_ops.bias_add(candidate0, _candidate_bias)
        c = math_ops.tanh(candidate)
        u1 = (1 - att_score) * u
        new_h = u1 * state + (1 - u1) * c
        sess.run(tf.global_variables_initializer())
        loss = tf.reduce_mean(new_h - 1)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = optimizer.compute_gradients(loss)
        g_candidate = sess.run((1 - u1)*(1-c*c))*sess.run(tf.gradients(loss, new_h))
        print('g_candidate', np.squeeze(g_candidate))
        print('loss/candidate', sess.run(tf.gradients(loss, candidate)))
        print('loss/candidate_bias', sess.run(tf.gradients(loss, _candidate_bias)))
        #g_candiate_kernel
        concat_input = sess.run(array_ops.concat([inputs, r_state], 1))
        print('array_ops.concat([inputs, r_state], 1)', concat_input)
        g_c_kernel = math_ops.matmul(np.transpose(concat_input) , g_candidate)
        print('g_c_kernel', sess.run(g_c_kernel))
        print('loss/_candidate_kernel', sess.run(tf.gradients(loss, _candidate_kernel)))
        #g_r_state
        print('loss/r_state', sess.run(tf.gradients(loss, r_state)))
        g_input_rstate = math_ops.matmul( np.squeeze(g_candidate),np.transpose(sess.run(_candidate_kernel)))
        g_input_candidate, g_r_state = array_ops.split(value=g_input_rstate, num_or_size_splits=2, axis=1)
        print('g_r_state', sess.run(g_r_state))
        
        #g_r
        print('loss/r', sess.run(tf.gradients(loss, r)))
        g_r = sess.run(g_r_state)*sess.run(state)
        print('g_r', g_r)
        #g_u
        g_u = sess.run(state-c)*sess.run(tf.gradients(loss, new_h))*sess.run(1-att_score)
        print('loss/u', sess.run(tf.gradients(loss, u)))
        print('g_u', g_u)
        #g_gate_inputs
        print('loss/gate_inputs', sess.run(tf.gradients(loss, gate_inputs)))
        g_gate_inputs = array_ops.concat([g_r, np.squeeze(g_u)], 1) * value * (1-value)
        print('g_gate_inputs', sess.run(g_gate_inputs))
        g_gate_inputs_state = math_ops.matmul(g_gate_inputs, np.transpose(sess.run(_gate_kernel)))
        g_inputs_gate, g_state_gate = array_ops.split(value=g_gate_inputs_state, num_or_size_splits=2, axis=1)
        print('g_inputs_gate', g_inputs_gate)
        #g_gate_kernel
        print('loss/gate_kernel', sess.run(tf.gradients(loss, _gate_kernel)))
        g_gate_kernel = math_ops.matmul(np.transpose(concat_input), g_gate_inputs)
        print('g_gate_kernel', sess.run(g_gate_kernel))
        #g_gate_bias
        print('loss/gate_bias', sess.run(tf.gradients(loss, _gate_bias)))
        g_gate_bias = tf.reduce_sum(g_gate_inputs, axis=0)
        print('g_gate_bias', sess.run(g_gate_bias))
        print('loss/inputs', sess.run(tf.gradients(loss, inputs)))
        g_input=g_inputs_gate + g_input_candidate
        print('g_inputs', sess.run(g_input))
        print('loss/attention', sess.run(tf.gradients(loss, att_score)))
        g_att_score = sess.run(tf.gradients(loss, new_h) * (state - c) * (-u))
        print('g_attention', g_att_score)

# augru_unit()


'''sigmoid反向：sigmoid(x)*(1-sigmoid(x))'''
def sigmoid_grads():
    x = tf.Variable([1.0, 1.0], dtype=tf.float32, name='x')
    y = math_ops.sigmoid(x)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(y))
        print(sess.run(tf.gradients(y, x)))
        print(sess.run(y*(1-y)))
# sigmoid_grads()

'''tanh的反向为：1-(tanh(x))的平方'''
def tanh_grads():
    x = tf.Variable([1.0, 1.0], dtype=tf.float32, name='x')
    y = math_ops.tanh(x)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(y))
        print(sess.run(tf.gradients(y, x)))


def mul_grads():
    x = tf.Variable([[11, 12], [13, 14]], dtype=tf.float32)
    y = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    z = math_ops.matmul(x, y)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('g_x', sess.run(tf.gradients(z, x)))
        print('g_y', sess.run(tf.gradients(z, y)))

def mul_dot_grads():
    x = tf.Variable([[3, 3], [4, 4]], dtype=tf.float32)
    y = tf.Variable([[11, 12], [21, 22]], dtype=tf.float32)
    b = tf.Variable([[0.1, 0.1], [0.1, 0.1]], dtype=tf.float32)
    z = x * y + b
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('g_x', sess.run(tf.gradients(z, x)))
        print('g_y', sess.run(tf.gradients(z, y)))
        print('g_b', sess.run(tf.gradients(z, b)))
        
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)
    grads_and_vars = optimizer.compute_gradients(z)
    train_op = optimizer.apply_gradients(grads_and_vars)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z_out, gvars = sess.run([z, grads_and_vars])
        print('z=', z_out)
        for var, value in zip(grads_and_vars, gvars):
            print(var, value)
        for var in tf.global_variables():
            print(var, sess.run(var))
        sess.run(train_op)
        for var in tf.global_variables():
            print(var, sess.run(var))


def norm_grads():
    x = tf.Variable([[1, 1], [1, 1]], dtype=tf.float32)
    y = tf.norm(x)
    g = tf.gradients(y, [x])
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([y, g]))

'''
reduce_mean的gradients=1/element_number。相当于把1平均分到了各个元素上。
'''


def reduce_mean_grads():
    x = tf.Variable([[1, 1], [2, 2], [4,5]], dtype=tf.float32)
    y = tf.reduce_mean(x -1 )
    g = tf.gradients(y, [x])
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('loss=', sess.run(y))
        print(sess.run(g))


def reduce_mean_grads2():
    x = tf.Variable([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)
    loss = tf.reduce_mean(x)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.2)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([train_op, loss, grads_and_vars]))

reduce_mean_grads()
# reduce_mean_grads2()
