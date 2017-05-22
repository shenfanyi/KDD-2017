

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data1 = np.genfromtxt('./data/oneweek1_0.csv', delimiter=',')
# data = data1[1:,1:]

batch_size = 4

# x_train = np.reshape(data[0:48,0:8],(4,12,8))
# y_train = np.reshape(data[0:48,-1],(4,12))

# print(x_train.shape)
# print(y_train.shape)

# x_test = np.reshape(data[48:60,0:8],(1,12,8))
# y_test = data[48:60,-1]



data_train_1_0_volume = pd.read_csv('data_from_weikai/train_1_0_volume.csv')

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
x_train = np.reshape(volx, [6,16])
x_train = np.transpose(x_train)
x_train = np.reshape(x_train, [16,6,1])
# print x_train

voly = vol([y1,y2,y3,y4,y5,y6])
# print volx
y_train = np.reshape(voly, [6,16])
y_train = np.transpose(y_train)
# print y_train




model = Sequential()
model.add(LSTM(50,
              activation='tanh',
              input_shape=(6, 1),
              batch_size=batch_size,
              return_sequences=True,
              stateful=True))
model.add(LSTM(50,
               return_sequences=False,
               stateful=True))
model.add(Dense(6))
model.compile(loss='mse', optimizer='rmsprop')


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=300,
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