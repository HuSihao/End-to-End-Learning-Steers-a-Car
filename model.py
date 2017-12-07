'''
You are going to implement the CNN model from paper 'End to End Learning for Self-Driving Cars'.
Write the model below.
'''
import tensorflow as tf

class CNN(object):
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial);

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, strides):
        return tf.nn.conv2d(x, W, strides, padding='VALID')

    def conv_layer(self,x,name,conv=(3, 3), strides=[1, 1, 1, 1], n_out=32, use_bias=True):
        W = self.weight_variable([conv[0], conv[1], x.get_shape()[-1].value, n_out])
        if use_bias:
            b = self.bias_variable([n_out])
            return tf.nn.relu(self.conv2d(x, W, strides=strides) + b,name=name)
        else:
            return tf.nn.relu(self.conv2d(x, W, strides=strides),name=name)

    def fc_layer(self,x,name,n_neurons, activation=tf.nn.relu, keep_prob=1.0):
        W = self.weight_variable([x.get_shape()[-1].value, n_neurons])
        b = self.bias_variable([n_neurons])
        h = activation(tf.matmul(x, W) + b)
        return tf.nn.dropout(h, keep_prob,name=name)

    def __init__(self,name):
        self.name = name;
        self.xs = tf.placeholder(tf.float32, [None, 66, 200, 3],name='xs')
        self.ys = tf.placeholder(tf.float32, [None, 1],name='ys')
        self.conv1_layer = self.conv_layer(self.xs,name='conv1_layer',conv=(5, 5), strides=[1, 2, 2, 1], n_out=24)
        self.conv2_layer = self.conv_layer(self.conv1_layer,name='conv2_layer', conv=(5, 5), strides=[1, 2, 2, 1], n_out=36)
        self.conv3_layer = self.conv_layer(self.conv2_layer,name='conv3_layer', conv=(5, 5), strides=[1, 2, 2, 1], n_out=48)
        self.conv4_layer = self.conv_layer(self.conv3_layer,name='conv4_layer', conv=(3, 3), strides=[1, 1, 1, 1], n_out=64)
        self.conv5_layer = self.conv_layer(self.conv4_layer,name='conv5_layer', conv=(3, 3), strides=[1, 1, 1, 1], n_out=64)
        self.conv5_layer_flat = tf.reshape(self.conv5_layer, [-1, 1152])
        self.full1_layer = self.fc_layer(self.conv5_layer_flat,name='full1_layer', n_neurons =100, activation=tf.nn.relu, keep_prob=1.0)
        self.full2_layer = self.fc_layer(self.full1_layer,name='full2_layer',  n_neurons=50, activation=tf.nn.relu, keep_prob=1.0)
        self.full3_layer = self.fc_layer(self.full2_layer,name='full3_layer', n_neurons=10, activation=tf.nn.relu, keep_prob=1.0)
        self.y_ = tf.matmul(self.full3_layer,self.weight_variable([10,1])+self.bias_variable([1]),name='y_')
        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.ys, self.y_)),name='loss')
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss,name='train_step')

