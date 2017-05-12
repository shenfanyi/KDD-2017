
# 1 handle data

import pandas as pd
import numpy as np

am = pd.read_csv('am.csv')
pm = pd.read_csv('pm.csv')

# print am.describe()
# print pm.iloc[0:3]

# print am.groupby('date').count()
# print am.date.value_counts().count()
# print pm.date.value_counts().count()

# print am[am.eval('date == "[2016-09-19"')]

am_date = am.date.unique()
am_date = np.delete(am_date,[21,13])
# print am_date

am = am[am.date.isin(am_date)]
am = am.loc[:,['volume','time_window','date']]

am_train = am[am.date.isin(am_date[0:22])].loc[:,'volume']
am_pred = am[am.date.isin(am_date[-7:])].loc[:,'volume']
# print am_train.count()
# print am_pred.count()

am_train = np.array(am_train)
am_train = np.reshape(am_train,[110,12])
# print am_train
am_pred = np.array(am_pred)
am_pred = np.reshape(am_pred,[35,6])
# print am_pred


xtime = [np.arange(1,7)]
# xtime = np.repeat(xtime,3,axis=0)



# 2 build model

import tensorflow as tf
from tensorflow.contrib import rnn

batch_size = 4
n_steps = 6
n_input = 1 
n_output = 1
data_length = 90

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_steps, n_output])

inputx = np.reshape(np.repeat(xtime,90,axis=0), [-1,6,1])
inputy1 = np.reshape(am_train[0:90,0:6], [-1,6,1])
inputy2 = np.reshape(am_train[0:90,6:12], [-1,6,1])

testx = np.reshape(np.repeat(xtime,20,axis=0), [-1,6,1])
testy1 = np.reshape(am_train[90:,0:6], [-1,6,1])
testy2 = np.reshape(am_train[90:,6:12], [-1,6,1])

# predx = np.reshape(am_train, [-1,6,1])

def RNN(x):
    x = tf.unstack(x, n_steps, 1)
    # x = tf.transpose(x,[1,0,2])
    # x = tf.reshape(x,[-1,n_input])
    # x = tf.split(x,n_steps)
    lstm_cell = rnn.BasicLSTMCell(n_output, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return outputs

pred = RNN(x)
pred1 = tf.reshape(pred, [n_steps,-1,n_output])
pred2 = tf.transpose(pred1, [1,0,2])


# cost = tf.contrib.losses.sigmoid_cross_entropy(pred2,y)
cost = tf.reduce_mean(tf.square(pred2 - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

accuracy = tf.reduce_mean(tf.square(tf.subtract(pred2, y)))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        index = np.random.choice(data_length, batch_size, replace=False)
        print 'index',index
        sess.run(optimizer, feed_dict={x:inputx[index], y:inputy1[index]})
        print 'cost:',sess.run(cost, feed_dict={x:inputx[index], y:inputy1[index]})
        print 'accu:',sess.run(accuracy, feed_dict={x:testx, y:testy1})
    # print 'pred:',sess.run(pred2,feed_dict={x:predx})

