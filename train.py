'''
You are going to train the CNN model here.

'''
import tensorflow as tf
import numpy as np
import os
from model import CNN
from load_data import LoadTrainBatch,LoadValBatch

LOGDIR = './save'
CKPT_FILE = './save/model.ckpt'
TRAIN_TENSORBOARD_LOG = './train_logs'
VAL_TENSORBOARD_LOG = './val_logs'

batch_size = 50
max_iter_num = 10000

cnn = CNN('cnn')
saver = tf.train.Saver()

train_summary = tf.summary.scalar("train_loss",cnn.loss)
val_summary = tf.summary.scalar("val_loss",cnn.loss)

train_summary_writer = tf.summary.FileWriter(TRAIN_TENSORBOARD_LOG, graph=tf.get_default_graph())
val_summary_writer = tf.summary.FileWriter(VAL_TENSORBOARD_LOG, graph=tf.get_default_graph())

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(max_iter_num):
        [x_train, y_train] = LoadTrainBatch(batch_size)
        cnn.train_step.run(feed_dict={cnn.xs: x_train, cnn.ys: y_train})
        if step % 10 ==0:
            x_val, y_val = LoadValBatch(batch_size)
            train_loss = cnn.loss.eval(feed_dict={cnn.xs: x_train, cnn.ys: y_train})
            val_loss = cnn.loss.eval(feed_dict={cnn.xs: x_val, cnn.ys: y_val})
            print('step %d, train loss_value:%g val loss:%g' % (step, train_loss, val_loss))
        if step %100 == 0:
            checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)