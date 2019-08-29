import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
#from sklearn.preprocessing import MinMaxScaler
import sys
import time

def predict_next(currSpeed, model):
    #currSpeed = [currSpeed] * 1
    #currSpeed = np.array(currSpeed)
    #currSpeed = np.reshape(currSpeed,(1,1,1))
    length = len(currSpeed)
    currSpeed = np.reshape(currSpeed,(length, 1, 1)) #Batch size corresponding to length of currSpeed == n 
    np.random.seed(7)
    #Scale test input X to range (0,1)
    # This is temporary and Not the Right scaling. The Right scaling factor is maximum Possible Speed
    maxSpeed = np.max(currSpeed)
    currSpeed = ((1.0)*(currSpeed))/maxSpeed
    #Predict
    nextSpeed = model.predict(currSpeed)
    nextSpeed = np.reshape(nextSpeed,(length,))
    #print(nextSpeed)
    maxSpeed = np.max(nextSpeed)
    nextSpeed = (10.0/maxSpeed) * (nextSpeed)
    nextSpeed = np.round(nextSpeed)
    #nextSpeed = ((1.0)*(nextSpeed))/maxSpeed
    #print(nextSpeed)
    #nextSpeed = map(lambda x: int(x*10), nextSpeed)
    nextSpeed = map(lambda x: int(x), nextSpeed)
    print(nextSpeed)
    return nextSpeed

"""
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# load model
model = load_model("lstmForSpeed.h5")
# invert predictions
testPredict = scaler.inverse_transform(testPredict)
"""
if __name__=="__main__":
    currSpeed = sys.argv[1]
    #Load saved model
    model = load_model("lstmForSpeed.h5")
    begin = time.time()
    nextSpeed = predict_next(currSpeed, model)
    predictTime = time.time()-begin
    print currSpeed, nextSpeed, predictTime
