

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

data1 = np.genfromtxt('./data/oneweek1_0.csv', delimiter=',')
data = data1[1:,1:]
# print(data.shape)
# print(data[1])

batch_size = 2
n_steps = 12
n_input = 8 
n_output = 1
data_length = 5


x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_steps, n_output])


def RNN(x):
    x = tf.unstack(x, n_steps, 1)
    # x = tf.transpose(x,[1,0,2])
    # x = tf.reshape(x,[-1,n_input])
    # x = tf.split(x,n_steps)
    lstm_cell = rnn.BasicLSTMCell(n_output, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return outputs

pred = RNN(x)
pred1 = tf.reshape(pred, [12,-1,n_output])
pred2 = tf.transpose(pred1, [1,0,2])


# cost = tf.contrib.losses.sigmoid_cross_entropy(pred2,y)
cost = tf.reduce_mean(tf.square(pred2 - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

# accuracy = tf.reduce_mean(tf.square(tf.subtract(pred2, y)))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    x_data = np.reshape(data[:,0:8],[-1,12,8])
    y_data = np.reshape(data[:,-1],[-1,12,1])

    for i in range(50):
        index = np.random.choice(data_length, batch_size, replace=False)
        print 'index',index
        sess.run(optimizer, feed_dict={x:x_data[index], y:y_data[index]})
        print sess.run(cost, feed_dict={x:x_data[index], y:y_data[index]})
