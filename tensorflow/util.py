import tensorflow as tf
from tensorflow.python.util import nest

graph=tf.compat.v1.Graph()

var=tf.Variable(([1,2],[3,4]))
fvar=nest.flatten(var)
op=tf.matmul(var,fvar)
print(var)
print(fvar)
with tf.compat.v1.Session() as sess:
    
    print(sess.run(op))
#     print(sess.run(fvar))