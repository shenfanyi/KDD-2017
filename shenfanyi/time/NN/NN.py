

import pandas as pd
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


tratime = pd.read_csv('time_1.csv')
weather = pd.read_csv('weather_training.csv')

def hour(x):
	return int(x.split(':')[0])

tratime.hour = tratime.hour.apply(hour)

# print tratime.hour

time = pd.merge(tratime, weather,how='inner',on=['date','hour'])
# print time.iloc[0,:]
# print pd.isnull(time).any()
# print time.isnull().values.any()

time = time[time.date.isin([
    '2016-07-26','2016-08-15','2016-08-16','2016-08-17','2016-08-18','2016-08-19','2016-08-20','2016-08-21','2016-08-29','2016-08-30','2016-08-31',
    '2016-09-01','2016-09-02','2016-09-03','2016-09-21','2016-09-28','2016-09-29','2016-09-30','2016-10-01',
    '2016-10-02','2016-10-03','2016-10-04','2016-10-05','2016-10-06','2016-10-07'
    ])]

# print time[time.eval('tollgate_id == 3 & intersection_id == "B"')]

train_x = time[time.eval('tollgate_id == 3 & intersection_id == "B"')].iloc[:300,:].loc[:,'hour':]
train_y = time[time.eval('tollgate_id == 3 & intersection_id == "B"')].iloc[:300,:].loc[:,'avg_travel_time']
test_x = time[time.eval('tollgate_id == 3 & intersection_id == "B"')].iloc[300:,:].loc[:,'hour':]
test_y = time[time.eval('tollgate_id == 3 & intersection_id == "B"')].iloc[300:,:].loc[:,'avg_travel_time']


trX, trY, teX, teY = np.array(train_x), np.reshape(train_y,(-1,1)), np.array(test_x), np.reshape(test_y,(-1,1))
print trX.shape,trY.shape

data_length = 300 #2168,2168,2168,2170,2169
batch_size = 100
epoch = 100

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h3, w_h2, w_h1, b1, b2, b3):
    # X = tf.nn.dropout(X, p_keep_input)
    h1 = tf.nn.relu(tf.matmul(X, w_h1) + b1)
    h1 = tf.nn.dropout(h1, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h1, w_h2) + b2)
    # h2 = tf.nn.dropout(h2, p_keep_hidden)
    h3 = tf.nn.elu(tf.matmul(h2, w_h3) + b3)
    return h3

X = tf.placeholder("float", [None, 10])
Y = tf.placeholder("float", [None, 1])

w_h1 = init_weights([10, 1000])
w_h2 = init_weights([1000, 1000])
w_h3 = init_weights([1000, 1])
b1 = init_weights([1000])+1000
b2 = init_weights([1000])+1000
b3 = init_weights([1])+1000


p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, w_h3, w_h2, w_h1, b1, b2, b3)


# cost = tf.contrib.losses.sigmoid_cross_entropy(py_x,Y) ## smaller 
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) ## be 0.00 all the time
# cost = tf.reduce_mean(tf.square(py_x - Y))  ##up,down,up,down, very large
cost = tf.reduce_mean(tf.abs(tf.div(tf.subtract(Y,py_x),Y/1.0))) 

# train_op = tf.train.AdamOptimizer().minimize(cost) 
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    lossall = []
    for i in range(epoch):
        for j in range(100):
            index = np.random.choice(data_length, batch_size, replace=False)
            sess.run(train_op, feed_dict={X: trX[index], Y: trY[index], p_keep_input: 0.8, p_keep_hidden: 0.8})
        loss = sess.run(cost, feed_dict={X: trX, Y: trY, p_keep_input: 1.0, p_keep_hidden: 1.0})
        accuracy = sess.run(cost, feed_dict={X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden: 1.0})
        print 'loss:',loss
        print 'accuracy:',accuracy
        # print sess.run(py_x, feed_dict={X: trX[0:25], p_keep_input: 1.0, p_keep_hidden: 1.0})
        lossall.append(loss)
    print sess.run(py_x, feed_dict={X: teX[0:25], p_keep_input: 1.0, p_keep_hidden: 1.0})
    print teY[0:25]
    plt.plot(lossall)
    plt.show()