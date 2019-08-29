import json
import numpy as np
from masterServerStatic import matrixMulKernelMaster
from masterServerStatic import encode_matrix
from masterServerStatic import encode_matrix_tp
import multiprocessing
import time
import os
from sklearn.model_selection import train_test_split

def computeProb(k, n, iteration, dataset, weights, partitions, execTimes):
    result, compute_time, communication_time, decode_time = matrixMulKernelMaster(iteration, dataset, execTimes)
    result = 1.0 / (1 + np.exp(-1*result))  
    return result, compute_time, communication_time, decode_time

def computeGrad(k, n, iteration, dataset, weights, partitions, execTimes):
    result, compute_time, communication_time, decode_time = matrixMulKernelMaster(iteration, dataset, execTimes)
    return result, compute_time, communication_time, decode_time

def transferWeights(wtFile, n):
    n = 4
    for i in range(1, n+1):
        cmd = "scp %s slave%d:%s" % (wtFile, i, wtFile)
        os.system(cmd)

def run(first=False, last=False):
    configs = json.load(open('config/config.json'))
    encoding = np.array(configs['matrixConfigs']['encoding'])
    CHUNKS = configs['chunks']
    execTimes = configs['execTimes']
    k, n = encoding.shape
    global X, y, X_tr, X_tr_tp, X_te, y_tr, y_te
    if first:
        X = np.loadtxt('/home/zhifeng/apps/hadoop/data/gisette_train.data', dtype=int)
        y = np.loadtxt('/home/zhifeng/apps/hadoop/data/gisette_train.labels', dtype=int)
        # some preprocessing
        X = np.hstack((np.ones((X.shape[0], 1)), X)) 
        y[y < 0] = 0
        y = y.reshape(y.shape[0],1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=7.0/700.0)
        print X_tr.shape, y_tr.shape
        common_div = k * CHUNKS * n
        dimension = int(np.ceil(float(X_tr.shape[1])/common_div)) * common_div
        X_tr_tp = np.vstack((X_tr.T, np.zeros((dimension-X_tr.shape[1], X_tr.shape[0]))))
        print X_tr_tp.shape
    weights = np.random.rand(X.shape[1], 1)
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
    iterations = 60 
    losses = []
    sstart_time = time.time()
    wtFile = '/home/zhifeng/apps/static/data/weights.mat'
    resultFile = open('resultsLR.out', 'a')
    for iteration in range(iterations):
        start_time = time.time()
        np.savetxt(wtFile, weights, fmt='%.10f')
        transferWeights(wtFile, n)
        communication_time += time.time() - start_time
        if not iteration:
            cmd = 'bash /home/zhifeng/apps/launchSlavesServer.sh %s N' % os.getcwd()
            time.sleep(10)
            sstart_time = time.time()
            resultFile.write(str(execTimes)+'\n')

        learning_rate = 1.0 / (iteration + 1.0)
        if first:
            prob, computeT, communicationT, decodeT = computeProb(k, n, iteration, X_tr, weights, partitions, execTimes)
        else:
            prob, computeT, communicationT, decodeT = computeProb(k, n, iteration+1, X_tr, weights, partitions, execTimes)
        compute_time += computeT
        communication_time += communicationT
        decode_time += decodeT
        start_time = time.time()
        errors = y_tr - prob
        serial_time += time.time() - start_time
        start_time = time.time()
        np.savetxt(wtFile, errors, fmt='%.10f')
        transferWeights(wtFile, n)
        communication_time += time.time() - start_time
        gradient, computeT, communicationT, decodeT = computeGrad(k, n, iteration+1, X_tr_tp, errors, partitions_tp, execTimes)
        compute_time += computeT
        communication_time += communicationT
        decode_time += decodeT
        start_time = time.time()
        gradient = gradient[weights.shape[0], :]
        weights += learning_rate * (gradient - reg * weights)
        cond_prob = prob
        cond_prob[y_tr < 1] *= -1
        cond_prob[y_tr < 1] += 1
        loss = np.log(cond_prob.sum()) - 0.5 * reg * (weights * weights).sum()
        losses.append(loss)
        serial_time += time.time() - start_time
        if iteration > 0 and iteration % 15 == 0:
            resultFile.write('compute_time = %f seconds\n' % compute_time)
            resultFile.write('communication_time = %f seconds\n' % communication_time)
            resultFile.write('decode_time = %f seconds\n' % decode_time)
            resultFile.write('serial_time = %f seconds\n' % serial_time)
            compute_time = 0
            communication_time = 0
            decode_time = 0
            serial_time = 0
            execTimes[(iteration/15)-1] *= 20
            resultFile.write('\n')
            resultFile.write(str(execTimes)+'\n')

    end_time = time.time()
    pred = X_te.dot(weights)
    pred[pred < 0] = 0
    pred[pred >= 0] = 1
    print 'test error is %f' % (np.abs(y_te - pred).sum() / y_te.shape[0])
    print losses
    print 'total execution time is %f seconds' % (end_time - sstart_time)
    print 'distEncode_time = %f seconds' % distEncode_time
    resultFile.write('compute_time = %f seconds\n' % compute_time)
    resultFile.write('communication_time = %f seconds\n' % communication_time)
    resultFile.write('decode_time = %f seconds\n' % decode_time)
    resultFile.write('serial_time = %f seconds\n' % serial_time)
    resultFile.write('\n')
    if last:
        cmd = "bash /home/zhifeng/apps/killSlavesDistributed.sh"
        os.system(cmd)
    resultFile.close() 

def main():
    np.random.seed(1351)
    total = 3
    for trial in range(total):
        if not trial:
            run(first=True)
        elif trial == total - 1:
            run(last=True)
        else:
            run()

if __name__ == '__main__':
    main()
    
    
