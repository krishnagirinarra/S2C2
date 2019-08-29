import json
import numpy as np
from poly_masterServerStatic import matrixMulKernelMaster
from poly_masterServerStatic import encode_matrix
from poly_masterServerStatic import encode_matrix_tp
import multiprocessing
import time
import os
from sklearn.model_selection import train_test_split

def computeScore(k, n, iteration, dataset, weights, partitions, execTimes):
    result, compute_time, communication_time, decode_time = matrixMulKernelMaster(iteration, dataset, execTimes)
    return result, compute_time, communication_time, decode_time

def computeGrad(k, n, iteration, dataset, weights, partitions, execTimes):
    result, compute_time, communication_time, decode_time = matrixMulKernelMaster(iteration, dataset, execTimes)
    return result, compute_time, communication_time, decode_time

def transferWeights(wtFile, n):
    cmd = "parallel --no-notice :::"
    configs = json.load(open('config/config.json'))
    for i in range(1, n+1):
        slaveIP = configs['slaveConfigs']['slave'+str(i)]['IP']
        cmd += " \"scp -P 5000 %s %s:%s\"" % (wtFile, slaveIP, wtFile)
    os.system(cmd)

def run(first=False, last=False):
    configs = json.load(open('config/config.json'))
    encoding = np.array(configs['matrixConfigs']['encoding'])
    CHUNKS = configs['chunks']
    execTimes = configs['execTimes']
    k, n = 9, 12
    global X, y, X_tr, X_tr_tp, X_te, y_tr, y_te
    dim = 6000
    if first:
        X_tr = np.zeros((dim, 2))
    wt_rows = dim # c=6000, r=6000
    weights = np.random.rand(dim, 1)#in place of Diagonal values
    partitions = []
    partitions_tp = []
    distEncode_time = 0
    compute_time = 0
    communication_time = 0
    decode_time = 0
    serial_time = 0

    start_time = time.time()
    distEncode_time += time.time() - start_time
    reg = 0.5
    iterations = 15
    losses = []
    sstart_time = time.time()
    wtFile = '/home/zhifeng/apps/static/data/weights.mat'
    resultFile = open('resultsSVM.out', 'a')
    for iteration in range(iterations):
        start_time = time.time()
        np.savetxt(wtFile, weights, fmt='%.10f')
        transferWeights(wtFile, n)
        transferTime = time.time() - start_time
        communication_time += transferTime
        print 'transfer time is %f seconds' % (transferTime)
        if not iteration:
            cmd = 'bash /home/zhifeng/apps/launchSlavesServer.sh %s N' % os.getcwd()
            time.sleep(10)
            sstart_time = time.time()
            resultFile.write(str(execTimes)+'\n')

        learning_rate = 1.0 / (iteration + 1.0)
        if first:
            score, computeT, communicationT, decodeT = computeScore(k, n, iteration, X_tr, weights, partitions, execTimes)
        else:
            score, computeT, communicationT, decodeT = computeScore(k, n, iteration+1, X_tr, weights, partitions, execTimes)
        compute_time += computeT
        communication_time += communicationT
        decode_time += decodeT
        start_time = time.time()
        serial_time += time.time() - start_time
        weights -= learning_rate * (reg * weights)
        if iteration > 0 and (0+iteration) % 15 == 0:
            resultFile.write('compute_time = %f seconds\n' % compute_time)
            resultFile.write('communication_time = %f seconds\n' % communication_time)
            resultFile.write('decode_time = %f seconds\n' % decode_time)
            resultFile.write('serial_time = %f seconds\n' % serial_time)
            compute_time = 0
            communication_time = 0
            decode_time = 0
            serial_time = 0
            resultFile.write('\n')
            resultFile.write(str(execTimes)+'\n')
            
    end_time = time.time()
    print 'total execution time is %f seconds' % (end_time - sstart_time)
    print 'distEncode_time = %f seconds' % distEncode_time
    resultFile.write('\n')
    if last:
        cmd = "bash /home/zhifeng/apps/killSlavesDistributed.sh"
        os.system(cmd)
    resultFile.close() 

def main():
    np.random.seed(1351)
    run(first=True, last=True)

if __name__ == '__main__':
    main()
    
    
