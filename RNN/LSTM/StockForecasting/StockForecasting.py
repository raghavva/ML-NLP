import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_csv('AAPL.CSV')
df = pd.read_csv('NIFTY 50.CSV')
#print(df.head(2))

#print(df.columns)

df1 = df.reset_index()['Close']  #reset index to make sure multi indexing does not happen
print(df1.shape)
plt.plot(df1)
#plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1)) #converts to an 2Darray with values b/w 0-1 .. applies z-score internally 
print(df1.shape)

#split data to test and train , since it is time series data , we need to take test data as the most recent one
# stock value depends on previous day's close 

train_size = int(len(df1)*0.70)
test_size = len(df1)-train_size
train_data, test_data = df1[0:train_size,:], df1[train_size:len(df1),:]
print(train_size,test_size)


#we try to create moving averages and store them into arrays
def create_dataset(dataset, time_counter = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_counter-1):
        a = dataset[i:(i+time_counter), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_counter, 0])    #the actual y-value is stored in y_train
    return np.array(dataX), np.array(dataY)

time_counter = 100
X_train, y_train = create_dataset(train_data, time_counter)
X_test, y_test = create_dataset(test_data, time_counter)

#print(X_train.shape)   #each row of X contains data of 100 previous days

#time series data using LSTM needs the data be 3 dimensional [samples,timesteps,features]
#here we are considering only 1 feature and hence features = 1

features = 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)


#LSTM Model
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Input

#model = Sequential([
#    Input(shape=(time_counter, 1)),  # Define input shape since we used 100 as time counter and 1 feature
#   LSTM(50, return_sequences = True),
#    LSTM(50),
#    Dense(1,activation = 'relu')
#])

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (100,1)))
model.add(LSTM(50, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))


model.compile(loss = 'mean_squared_error', optimizer = 'adam')

print(model.summary())

model_history = model.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose = 1)

#checking performance
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#get back the value to original form , hence perform inverse_transform
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

#RMSE performance
import math
from sklearn.metrics import mean_squared_error
print(math.sqrt(mean_squared_error(y_train,train_predict)))
print("/n")
print(math.sqrt(mean_squared_error(y_test,test_predict)))

### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1), label = 'Actual curve')
plt.plot(trainPredictPlot, label = 'Training prediction')
plt.plot(testPredictPlot, label = 'Testing prediction')
plt.legend()
plt.show()