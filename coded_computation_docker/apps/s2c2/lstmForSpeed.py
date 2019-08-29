# Adapted code from LSTM for international airline passengers problem with regression framing
import numpy
#import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
import sys

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('speedData.txt', engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
#print dataset.shape
#print numpy.max(dataset)
#sys.exit(0)
# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
sys.exit(0)
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
# save model
model.save("lstmForSpeed.h5")
# load model
model = load_model("lstmForSpeed.h5")
# make predictions
trainPredict = model.predict(trainX)
start = time.time()
testPredict = model.predict(testX)
end = time.time()
print("Time for test prediction", end-start)
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.4f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.4f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
invData = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
invData[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
#invData = scaler.inverse_transform(dataset)
#invData = invData[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :]
#for r,p in zip(invData,testPredict):
#    print(r,p)
#plt.plot(invData)
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.plot(testY[0], 'b', testPredict, 'r')
#plt.show()
