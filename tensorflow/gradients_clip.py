import tensorflow as tf
import numpy as np

grads = np.array([[0.9,0.8,0.7],[0.9,0.8,0.7]])
grads = tf.Variable([[0.9,0.8,0.7],[0.9,0.8,0.7]])
with  tf.compat.v1.Session() as sess:
    grads = tf.clip_by_global_norm(grads,5.0)
    grads = tf.clip_by_value(grads,5.0,10.0)
    print(sess.run(grads))