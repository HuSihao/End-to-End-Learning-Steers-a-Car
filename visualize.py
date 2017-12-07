import tensorflow as tf
import numpy as np
import os
from model import CNN
from load_data import LoadTrainBatch,LoadValBatch

ckpt = tf.train.get_checkpoint_state('./save/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    graph = tf.get_default_graph()

    conv1_layer = graph.get_operation_by_name('conv1_layer')
    conv2_layer = graph.get_operation_by_name('conv2_layer')
    conv3_layer = graph.get_operation_by_name('conv3_layer')
    conv4_layer = graph.get_operation_by_name('conv4_layer')
    conv5_layer = graph.get_operation_by_name('conv5_layer')
    loss = graph.get_operation_by_name('loss')
    xs = graph.get_operation_by_name('xs')


    [x, y] = LoadTrainBatch(1)

    print('step %d, train loss_value:%g val loss:%g' % (cnn.loss.eval(feed_dict={cnn.xs: x, cnn.ys: y})))
