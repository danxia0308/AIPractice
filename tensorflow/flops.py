import tensorflow as tf

def stats_graph(graph):
    flops=tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('flops={}'.format(flops.total_float_ops))
    print('Trainable params:{}'.format(params.total_parameters))
#     print(flops)

'''
tf.random_normal_initializer    FLOPs为参数量*2
矩阵相加的FLOPs为shape[0]*...*shape[n-1]
矩阵m*n和n*l相乘的FLOPs为m*n*l*2。每次加乘=一次乘+一次加，即2次浮点运算。
'''
def cal_flops1():
    with tf.Graph().as_default() as graph:
        A=tf.get_variable(initializer=tf.random_normal_initializer(dtype=tf.float32), shape=(4,4), name='A')
        B=tf.get_variable(initializer=tf.random_normal_initializer(dtype=tf.float32), shape=(4,4), name='B')
#         C=tf.add(A,B,name='output')
        C=tf.matmul(A,B,name='output')
        stats_graph(graph)

def get_model_graph():
    checkpoint_dir='/Users/chendanxia/sophie/checkpoint/temp/'
#     checkpoint_dir='/Users/chendanxia/sophie/checkpoint/shufflenet/shufflenet_0.784/'
    state=tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        saver=tf.train.import_meta_graph(state.model_checkpoint_path+'.meta', clear_devices=True)
        saver.restore(sess, state.model_checkpoint_path)
        graph=tf.get_default_graph()
        return graph

def cal_flops2():
    graph=get_model_graph()
    stats_graph(graph)

cal_flops2()