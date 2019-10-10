import tensorflow as tf





'''
变量在run之前，需要先run tf.global_variables_initializer()来初始化所有变量，或者run v.initializer来对单个变量初始化。
变量可以通过assign赋值来成为一个Tensor，其中可以用placeholder或者python数值来赋值。
'''
def variables():
    with tf.Session() as sess:
        v = tf.Variable([2],name='v')
        w = tf.Variable(2.0)
        print(v)
        print(w)
    #     sess.run(v.initializer)
        sess.run(tf.global_variables_initializer())
        v1,w1=sess.run([v,w])
        print(v1)
        print(w1)    
        
        w_a=w.assign(3)
        w1=sess.run(w_a)
        print(w1)
        
        w_p=tf.placeholder(dtype=tf.float32, shape=[], name='w_placeholder')
        w_a=w.assign(w_p)
        w1=sess.run(w_a, feed_dict={w_p:6})
        print(w1)

variables()

'''
eval只能用于tensor，等价于sess.run()
'''
def evals():
    a=tf.constant(45.0)
    b=tf.constant(15.0)
    c=tf.divide(a,b)
    with tf.Session() as sess:
        print(a.eval())
        print(c.eval())

