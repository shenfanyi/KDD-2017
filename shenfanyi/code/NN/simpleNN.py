import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('trainset_task2_version2.csv')
test = pd.read_csv('testset_task2_version2.csv')

train10 = train[train.eval('tollgate_id == 1 & direction == 0')]
train11 = train[train.eval('tollgate_id == 1 & direction == 1')]
train20 = train[train.eval('tollgate_id == 1 & direction == 0')]
train30 = train[train.eval('tollgate_id == 3 & direction == 0')]
train31 = train[train.eval('tollgate_id == 3 & direction == 1')]

test10 = test[test.eval('tollgate_id == 1 & direction == 0')]
test11 = test[test.eval('tollgate_id == 1 & direction == 1')]
test20 = test[test.eval('tollgate_id == 1 & direction == 0')]
test30 = test[test.eval('tollgate_id == 3 & direction == 0')]
test31 = test[test.eval('tollgate_id == 3 & direction == 1')]

train10_x = train10.iloc[:,-42:]
train11_x = train11.iloc[:,-42:]
train20_x = train20.iloc[:,-42:]
train30_x = train30.iloc[:,-42:]
train31_x = train31.iloc[:,-42:]

train10_y = train10.iloc[:,3]
train11_y = train11.iloc[:,3]
train20_y = train20.iloc[:,3]
train30_y = train30.iloc[:,3]
train31_y = train31.iloc[:,3]

test10_x = test10.iloc[:,-42:]
test11_x = test11.iloc[:,-42:]
test20_x = test20.iloc[:,-42:]
test30_x = test30.iloc[:,-42:]
test31_x = test31.iloc[:,-42:]

test10_y = test10.iloc[:,3]
test11_y = test11.iloc[:,3]
test20_y = test20.iloc[:,3]
test30_y = test30.iloc[:,3]
test31_y = test31.iloc[:,3]





trX, trY, teX, teY = np.array(train20_x), np.reshape(train20_y,(-1,1)), np.array(test20_x), np.reshape(test20_y,(-1,1))

data_length = 2168 #2168,2168,2168,2170,2169
batch_size = 1000
epoch = 100

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_o, w_h3, w_h2, w_h1, b0, b1, b2, b3):
    X = tf.nn.dropout(X, p_keep_input)
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1) + b1)
    h1 = tf.nn.dropout(h1, p_keep_hidden)
    h2 = tf.nn.sigmoid(tf.matmul(h1, w_h2) + b2)
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    h3 = tf.nn.sigmoid(tf.matmul(h2, w_h3) + b3)
    h3 = tf.nn.dropout(h3, p_keep_hidden)
    # h3 = tf.nn.sigmoid(tf.matmul(h2, w_h3))
    # h3 = tf.nn.dropout(h3, p_keep_hidden)
    # h3 = tf.nn.sigmoid(tf.matmul(h2, w_h3))
    # h3 = tf.nn.dropout(h3, p_keep_hidden) 
    return (tf.matmul(h3, w_o) + b0)

X = tf.placeholder("float", [None, 42])
Y = tf.placeholder("float", [None, 1])

w_h1 = init_weights([42, 42])
w_h2 = init_weights([42, 42])
w_h3 = init_weights([42, 42]) 
w_o = init_weights([42, 1])
b0 = init_weights([1])
b1 = init_weights([1])
b2 = init_weights([1])
b3 = init_weights([1])


p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, w_o, w_h3, w_h2, w_h1, b0, b1, b2, b3)


# cost = tf.contrib.losses.sigmoid_cross_entropy(py_x,Y) ## smaller 
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) ## be 0.00 all the time
# cost = tf.reduce_mean(tf.square(py_x - Y))  ##up,down,up,down, very large
cost = tf.reduce_mean(tf.abs(tf.div(tf.subtract(Y,py_x),Y/1.00)))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) 


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    lossall = []
    for i in range(epoch):
        for j in range(100):
            index = np.random.choice(data_length, batch_size, replace=False)
            # print 'index',index
            sess.run(train_op, feed_dict={X: trX[index], Y: trY[index], p_keep_input: 1.0, p_keep_hidden: 0.5})
            loss = sess.run(cost, feed_dict={X: trX[index], Y: trY[index], p_keep_input: 1.0, p_keep_hidden: 0.5})
            # sess.run(train_op, feed_dict={X: trX, Y: trY})
            # loss = sess.run(cost, feed_dict={X: trX, Y: trY})
        loss = sess.run(cost, feed_dict={X: trX, Y: trY, p_keep_input: 0.8, p_keep_hidden: 0.5})
        accuracy = sess.run(cost, feed_dict={X: teX, Y: teY, p_keep_input: 0.8, p_keep_hidden: 0.5})
        print 'loss:',loss
        print 'accuracy:',accuracy
        lossall.append(loss)
    print sess.run(py_x, feed_dict={X: teX[0:25], p_keep_input: 0.8, p_keep_hidden: 0.5})
    print teY[0:25]
    plt.plot(lossall)
    plt.show()