

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt

data1 = np.genfromtxt('./data/oneweek1_0.csv', delimiter=',')
data = data1[1:,1:]

batch_size = 1

x_train = np.reshape(data[0:48,0:8],(4,12,8))
y_train = np.reshape(data[0:48,-1],(4,12))

print(x_train.shape)
print(y_train.shape)

x_test = np.reshape(data[48:60,0:8],(1,12,8))
y_test = data[48:60,-1]


model = Sequential()
model.add(LSTM(50,
               input_shape=(12, 8),
               batch_size=batch_size,
               return_sequences=True,
               stateful=True))
model.add(LSTM(50,
               return_sequences=False,
               stateful=True))
model.add(Dense(12))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(1000):
    print('Epoch', i, '/', 5)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
predicted_output = model.predict(x_train, batch_size=batch_size)
print(predicted_output.shape)


print('Plotting Results')
plt.subplot(2, 1, 1)
plt.plot(np.transpose(y_train))
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(np.transpose(predicted_output))
plt.title('Predicted')
plt.show()