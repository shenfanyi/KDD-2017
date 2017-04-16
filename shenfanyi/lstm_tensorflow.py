

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = np.genfromtxt('./data/oneweek1_0.csv', delimiter=',')
data = data1[1:,1:]
#print data.shape


learning_rate = 0.001
training_iters = 500
batch_size = 1
display_step = 10

n_input = 8 
n_steps = 60 
n_hidden = 20
n_classes = 1 
 

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_steps,n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)


cost = tf.square(tf.subtract(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

accuracy = tf.square(tf.subtract(pred, y))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x = data[:,0:8]
        batch_y = data[:,-1]
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape((batch_size, n_steps,n_classes))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


