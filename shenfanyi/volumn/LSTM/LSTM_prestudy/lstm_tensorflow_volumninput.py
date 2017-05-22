
import pandas as pd
import numpy as np

data_train_1_0_volume = pd.read_csv('data_from_weikai/train_1_0_volume.csv')
# print 'train1_0',data_train_1_0_volume.shape

# data_train_1_1_volume = pd.read_csv('train_1_1_volume.csv')
# print 'train1_1',data_train_1_1_volume.shape

# data_train_2_0_volume = pd.read_csv('train_2_0_volume.csv')
# print 'train2_0',data_train_2_0_volume.shape

# data_train_3_0_volume = pd.read_csv('train_3_0_volume.csv')
# print 'train3_0',data_train_3_0_volume.shape

# data_train_3_1_volume = pd.read_csv('train_3_1_volume.csv')
# print 'train3_1',data_train_1_0_volume.shape

# print '------------------------'
# # print data_train_1_0_volume.iloc[0]
# # print data_train_1_0_volume.info()
# print '------------------------'
# # print data_train_1_0_volume['date'].dtype
# print '------------------------'
# print data_train_1_0_volume.iloc[0:2]


for i in ['[06:00:00,06:20:00)','[06:20:00,06:40:00)','[06:40:00,07:00:00)',
'[07:00:00,07:20:00)','[07:20:00,07:40:00)','[07:40:00,08:00:00)',
'[08:00:00,08:20:00)','[08:20:00,08:40:00)','[08:40:00,09:00:00)',
'[09:00:00,09:20:00)','[09:20:00,09:40:00)','[09:40:00,10:00:00)',
'[15:00:00,15:20:00)','[15:20:00,15:40:00)','[15:40:00,16:00:00)',
'[16:00:00,16:20:00)','[16:20:00,16:40:00)','[16:40:00,17:00:00)',
'[17:00:00,17:20:00)','[17:20:00,17:40:00)','[17:40:00,18:00:00)',
'[18:00:00,18:20:00)','[18:20:00,18:40:00)','[18:40:00,19:00:00)']:
	a = data_train_1_0_volume[data_train_1_0_volume.time_window==i ]
	b = a[a.weekday.isin([1,2,3,4,5])]
	# print b.volume.count()

# print data_train_1_0_volume.time_window.unique()

# print data_train_1_0_volume[data_train_1_0_volume.time_window=='[09:00:00,09:20:00)']

weekdaydata = data_train_1_0_volume[data_train_1_0_volume.weekday.isin([1,2,3,4,5])]
weekdaydata = weekdaydata[weekdaydata.holiday==False]
x1 = weekdaydata[weekdaydata.time_window=='[06:00:00,06:20:00)']
x2 = weekdaydata[weekdaydata.time_window=='[06:20:00,06:40:00)']
x3 = weekdaydata[weekdaydata.time_window=='[06:40:00,07:00:00)']
x4 = weekdaydata[weekdaydata.time_window=='[07:00:00,07:20:00)']
x5 = weekdaydata[weekdaydata.time_window=='[07:20:00,07:40:00)']
x6 = weekdaydata[weekdaydata.time_window=='[07:40:00,08:00:00)']

y1 = weekdaydata[weekdaydata.time_window=='[08:00:00,08:20:00)']
y2 = weekdaydata[weekdaydata.time_window=='[08:20:00,08:40:00)']
y3 = weekdaydata[weekdaydata.time_window=='[08:40:00,09:00:00)']
y4 = weekdaydata[weekdaydata.time_window=='[09:00:00,09:20:00)']
y5 = weekdaydata[weekdaydata.time_window=='[09:20:00,09:40:00)']
y6 = weekdaydata[weekdaydata.time_window=='[09:40:00,10:00:00)']

def vol(x):
	y = []
	for i in x:
		a = i.volume
		# print a.count()
		y.append(a)
	return y

volx = vol([x1,x2,x3,x4,x5,x6])
# print volx
inputx = np.reshape(volx, [6,16])
inputx = np.transpose(inputx)
inputx = np.reshape(inputx, [16,6,1])
# print inputx

voly = vol([y1,y2,y3,y4,y5,y6])
# print volx
inputy = np.reshape(voly, [6,16])
inputy = np.transpose(inputy)
inputy = np.reshape(inputy, [16,6,1])
# print inputy




import tensorflow as tf
from tensorflow.contrib import rnn

batch_size = 4
n_steps = 6
n_input = 1 
n_output = 1
data_length = 16

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
pred1 = tf.reshape(pred, [n_steps,-1,n_output])
pred2 = tf.transpose(pred1, [1,0,2])


# cost = tf.contrib.losses.sigmoid_cross_entropy(pred2,y)
cost = tf.reduce_mean(tf.square(pred2 - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

# accuracy = tf.reduce_mean(tf.square(tf.subtract(pred2, y)))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        index = np.random.choice(data_length, batch_size, replace=False)
        print 'index',index
        sess.run(optimizer, feed_dict={x:inputx[index], y:inputy[index]})
        print 'cost:',sess.run(cost, feed_dict={x:inputx[index], y:inputy[index]})


