import tensorflow as tf

def my_average_gradients(tower_grads, sess): 
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None: 
                expanded_g = tf.expand_dims(g, 0) 
                grads.append(expanded_g)

        if len(grads) ==0:
            continue
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        print('grad=',sess.run(grad))
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# vars=tf.Variable(['a','b','c','d'])
# grads1=tf.Variable([1.0,2,3,4,5])
# grads2=tf.Variable([6.0,7,8,9,10])
vars=['a','b','c','d']
grads1=[1.0,2,3,4,5]
grads2=[6.0,7,8,9,10]
grads1=[[1.0,1],[2,2],3,4,5]
grads2=[[6.0,6],[7,7],8,9,10]
a=tf.Variable('a',name='a')
b=tf.Variable(1,name='b')
c=tf.Variable(1,name='c')
d=tf.Variable(1,name='d')
# vars=tf.placeholder(dtype=tf.uint8,shape=(1,))
# grads1=tf.placeholder(dtype=tf.float32,shape=(1,))
# grads2=tf.placeholder(dtype=tf.float32,shape=(1,))
grads_list=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#     grads1=sess.run(grads1)
#     vars=sess.run(vars)
#     grads2=sess.run(grads2)
    grads_list.append(list(zip(grads1,vars)))
    grads_list.append(list(zip(grads2,vars)))
    print(sess.run(a))
    print(grads_list)
    average_grads=my_average_gradients(grads_list,sess)
    print(sess.run([average_grads]))
    print(average_grads)