import numpy as np
import json
from keras.models import load_model

configs = json.load(open('/home/zhifeng/apps/s2c2/config/config.json'))
encoding = np.array(configs['matrixConfigs']['encoding'])
k, n = encoding.shape
execTimes = configs['execTimes']

directory="/home/zhifeng/apps/speeds/"
slaveLengths = np.ones(n)
slaveExecTimes = np.ones(n)
histories = {}
forecasts = {}
arima_models = {}
#histLen = 20
histLen = 5
p = 1
d = 1
q = 1
model = load_model("lstmForSpeed.h5")
