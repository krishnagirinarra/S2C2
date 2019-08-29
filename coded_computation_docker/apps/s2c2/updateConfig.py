import os
import json
import numpy as np

newIps = os.environ['NODE_IPS'].split(':')
numSlaves = int(os.environ['TOTAL_SLAVES'])

configs = json.load(open('config/config.json'))
configs['masterConfigs']['IP'] = newIps[0]
for i in range(1, numSlaves):
    configs['slaveConfigs']['slave'+str(i)]['IP'] = newIps[i]

K = 7
N = numSlaves - 1
gMatrix = np.ones((K, N))
gMatrix[0:K, 0:K] = np.identity(K)

for j in range(K+1, N):
  for i in range(K):
    if not i:
      gMatrix[i,j] = 1
    else:
      gMatrix[i,j] = gMatrix[i,j-1] + gMatrix[i-1,j]
print gMatrix.astype(int)
configs['matrixConfigs']['encoding'] = gMatrix.astype(int).tolist()

json.dump(configs, open('config/config.json', 'w'))
