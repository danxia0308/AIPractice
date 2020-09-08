import numpy as np
import tensorflow.compat.v1 as tf
from Acclip import AcclipOptimizer

tf.disable_v2_behavior() 
np.random.seed(3)
tf.random.set_random_seed(3)

# x_vals = np.random.normal(1, 0.1, 100)
# y_vals = np.repeat(10.0, 100)
x_vals = 10*np.random.rand(100)
y_vals = 2*x_vals+3#+np.random.rand(100)
# print(x_vals)
# print(y_vals)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype= tf.float32)

A = tf.Variable(tf.random_normal(shape=[1]))
B = tf.Variable(tf.random_normal(shape=[1]))
my_output = tf.multiply(x_data, A)+B

loss = tf.square(my_output - y_target)

sess = tf.Session()
global_step = tf.placeholder(shape=[], dtype=tf.float32)
# global_step = tf.Variable(0.0, dtype=tf.float32, name="global_step", trainable=False)
# opt="Adam"
opt="SGD"
opt="Acclip"
if opt=="Adam":
    my_opt = tf.train.AdamOptimizer(learning_rate=0.8)
    grads_and_vars = my_opt.compute_gradients(loss)
#     print("grads_and_vars",grads_and_vars)
#     grads, vars = zip(*grads_and_vars)
#     print('grads',grads)
#     grads = tf.clip_by_global_norm(grads, 1)
#     print('grads',grads)
#     grads_and_vars = [(grad, var) for grad in grads[0] for var in vars]
#     print("grads_and_vars",grads_and_vars)
#     print('grads',grads)
#     print('vars',vars)
    train_step = my_opt.apply_gradients(grads_and_vars)
elif opt == "SGD":
    my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads_and_vars = my_opt.compute_gradients(loss)
#     grads, vars = zip(*grads_and_vars)
#     grads = tf.clip_by_global_norm(grads, 1)
#     grads_and_vars = [(grad, var) for grad in grads[0] for var in vars]
    train_step = my_opt.apply_gradients(grads_and_vars)
else:
    my_opt = AcclipOptimizer(learning_rate=0.8)
    grads_and_vars = my_opt.compute_gradients_adam(loss,global_step)
    train_step = my_opt.apply_gradients_adam(grads_and_vars)
# train_step,m_and_vs = my_opt.apply_gradients2(grads_and_vars)

# my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train_step = my_opt.minimize(loss)



init=tf.global_variables_initializer()  
#需要在train_step后init，因为Adam在这之前有自己的变量需要初始化。
sess.run(init)
g_vars = tf.trainable_variables()


var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print("var_list",var_list)
names=my_opt.get_slot_names()
print("slot_names",names)
print("slot_dict for m=",my_opt._slot_dict('m'))
# for var in var_list:
#     for name in names:
#         print(var,name,my_opt.get_slot(var, name), sess.run(my_opt.get_slot(var, name)))
#     for name in my_opt._non_slot_variables():
#         print(name, sess.run(name))
for var in my_opt._non_slot_variables():
    print(var,sess.run(var))
for i in range(100):#0到100,不包括100
    # 随机从样本中取值
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    #损失函数引用的placeholder(直接或间接用的都算), x_data使用样本rand_x， y_target用样本rand_y
#     grads,_,m_and_vs_value=sess.run([grads_and_vars,train_step,m_and_vs], feed_dict={x_data: rand_x, y_target: rand_y})
#     print('m_and_vs',m_and_vs_value)
#     print("    g_vars",g_vars)
#     print("    g_vars",sess.run(g_vars))
#     print("    non_slot_var",my_opt._non_slot_variables())
#     for var in my_opt._non_slot_variables():
#         print(var,sess.run(var))
#     for var in var_list:
#         m = sess.run(my_opt.get_slot(var, "m"), feed_dict={x_data: rand_x, y_target: rand_y})
#         v = sess.run(my_opt.get_slot(var, "v"), feed_dict={x_data: rand_x, y_target: rand_y})
#         print("    m=",m," v=",v)
    lastA=sess.run(A)
    lastB=sess.run(B)
    grads,_,loss_value=sess.run([grads_and_vars,train_step, loss], feed_dict={x_data: rand_x, y_target: rand_y, global_step:(i//10)})
#     print('m_and_vs',sess.run(m_and_vs))
    #打印
    if i%1==0:
        A_value=sess.run(A)
        B_value=sess.run(B)
        A_change=A_value-lastA
        B_change=B_value=lastB
        print("step:{},loss={},A={},B={},A_change={},B_change={}".format(i,loss_value, A_value,B_value,A_change,B_change))
#     for var in my_opt._non_slot_variables():
#         print(var,sess.run(var))
#         print('step: ' + str(i) + ' A = ' + str(sess.run(A))+" B = " + str(sess.run(B)))
#         print('A_change={}, B_change={}'.format((sess.run(A)-lastA),(sess.run(B)-lastB)))
#         print('    loss: ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y, global_step:i})))
#         print("    grads",grads)
#         print("    g_vars",g_vars)
#         print("    g_vars",sess.run(g_vars))
#         for var in var_list:
#             m = sess.run(my_opt.get_slot(var, "m"), feed_dict={x_data: rand_x, y_target: rand_y})
#             v = sess.run(my_opt.get_slot(var, "v"), feed_dict={x_data: rand_x, y_target: rand_y})
#             print("    m=",m," v=",v)
#         print("grads",grads)
#         print(sess.run(tf.clip_by_global_norm(grads[0], 5)))
#         print(sess.run(tf.clip_by_global_norm(grads[0],1)))
#         print(sess.run(tf.clip_by_value(grads[0][0], -300,10)))
#         break
        
        