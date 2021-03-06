from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[800]))
ds = ds.map(lambda x:[x,x])
ds = ds.repeat()

infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds, feed_name='infeed')
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

def body(x1, x2):
    d1 = x1 + x2
    d2 = x1 - x2
    outfeed = outfeed_queue.enqueue({'d1':d1, 'd2':d2})
    return outfeed

def my_net():
    r = loops.repeat(10,body, [], infeed_queue)
    return r

with scopes.ipu_scope('/device:IPU:0'):
    run_loop = ipu_compiler.compile(my_net, inputs=[])

dequeue_outfeed = outfeed_queue.dequeue()
# utils.move_variable_initialization_to_cpu()
config = utils.create_ipu_config()
config = utils.auto_select_ipus(config, 1)
utils.configure_ipu_system(config)

with tf.Session() as sess:
    sess.run(infeed_queue.initializer)
    sess.run(run_loop)
    print(sess.run(dequeue_outfeed))
    