import json
import numpy as np
from Optimal_masterServerStatic import matrixMulKernelMaster
from Optimal_masterServerStatic import encode_matrix
from Optimal_masterServerStatic import encode_matrix_tp
import multiprocessing
import time
import os
from sklearn.model_selection import train_test_split
import sys
import csv
from collections import defaultdict
from pandas import DataFrame
from sklearn import preprocessing
import mkl

mkl.set_num_threads(1)
##############################################################################################################
#### Implementation of Laplacian operator based Graph Signal Processing Algorithm to find n-Hop neighbors ####
##############################################################################################################
def loadMatrix():
    L = np.loadtxt('/home/krishna/finalApps/data/adj_matrix')
    X,Y = L.shape
    if (X!=Y):
        print("Input matrix is not a square matrix")
        sys.exit(0)
    M = np.zeros((4*X, 4*Y))
    #Replicating L many times to create a larger square Matrix
    M[0:X, 0:Y] = L
    M[0:X, Y:2*Y] = L
    M[X:2*X, 0:2*Y] = M[0:X, 0:2*Y]
    M[0:2*X, 2*Y:4*Y] = M[0:2*X, 0:2*Y]
    M[2*X:4*X, 0:4*Y] = M[0:2*X, 0:4*Y]
    loc_NaNs = np.isnan(M)
    M[loc_NaNs] = 0
    return M

def computeDegreeMatrix(A):
    # Find degree of each vertex
    # Create a diagonal matrix with degree of each vertex on its main diagonal position
    V = np.sum(A,axis=1)
    D = np.diag(V)
    return D 

def computeProb(k, n, iteration, dataset, weights, partitions, execTimes):
    result, compute_time, communication_time, decode_time = matrixMulKernelMaster(iteration, dataset, execTimes)
    return result, compute_time, communication_time, decode_time

def transferWeights(wtFile, n):
    n = 4
    for i in range(1, n+1):
        cmd = "scp %s slave%d:%s" % (wtFile, i, wtFile)
        os.system(cmd)

def run(first=False, last=False, resultFileName='resultGSP.out'):
    configs = json.load(open('config/config.json'))
    encoding = np.array(configs['matrixConfigs']['encoding'])
    CHUNKS = configs['chunks']
    execTimes = configs['execTimes']
    k, n = encoding.shape
    common_div = k * CHUNKS * n
    #Load matrix data
    M = loadMatrix()
    print "shape of M:", M.shape
    print M.sum(axis=0)
    N = int(np.floor(float(M.shape[0])/common_div)) * common_div
    M = M[:N, :N]
    D = computeDegreeMatrix(M)
    print "shape of D", D.shape
    L = D - M
    print "shape of L", L.shape
    #Definition of signal x. "pr" is x. It is initialized to random values. 
    pr = 1 + np.random.rand(N,1)
    pr = pr/np.linalg.norm(pr) #L2 norm
    # Store the g(x) = x + Lx + L^2 x.. 
    sum_Lx = np.zeros((N,1))
    last_pr = np.zeros((N,1))
    
    distEncode_time = 0
    compute_time = 0
    communication_time = 0
    decode_time = 0
    serial_time = 0
    start_time = time.time()
    partitions = []
    if first:
        partitions = encode_matrix(L, encoding)
    distEncode_time += time.time() - start_time
    print 'encoding time', str(distEncode_time)
    iterations = 60
    sstart_time = time.time()
    wtFile = '/home/krishna/finalApps/static/data/weights.mat'
    resultFile = open(resultFileName, 'a')
    for iteration in range(iterations):
        sum_Lx += pr
        last_pr = pr
        start_time = time.time()
        np.savetxt(wtFile, pr, fmt='%.5f')
        transferWeights(wtFile, n)
        communication_time += time.time() - start_time
        if first and not iteration:
            cmd = 'bash /home/krishna/finalApps/Optimal_launchSlaves.sh %s N' % os.getcwd()
        if not iteration:    
            sstart_time = time.time()
            resultFile.write('\n')
            resultFile.write(str(execTimes)+'\n')
        if first:
            pr, computeT, communicationT, decodeT = computeProb(k, n, iteration, L, pr, partitions, execTimes)
        else:
            pr, computeT, communicationT, decodeT = computeProb(k, n, iteration+1, L, pr, partitions, execTimes)
        print("LOG:communication Time %s seconds:" % str(communicationT))
        print("compute_time is %s seconds" % str(computeT))
        compute_time += computeT
        communication_time += communicationT
        decode_time += decodeT
        start_time = time.time()
        
        serial_time += time.time() - start_time
        if iteration > 0 and (iteration+1) % 15 == 0:
            resultFile.write('compute_time = %f seconds\n' % compute_time)
            resultFile.write('communication_time = %f seconds\n' % communication_time)
            resultFile.write('decode_time = %f seconds\n' % decode_time)
            resultFile.write('serial_time = %f seconds\n' % serial_time)
            compute_time = 0
            communication_time = 0
            decode_time = 0
            serial_time = 0
            execTimes[((1+iteration)/15)-1] *= 20
            resultFile.write('\n')
            resultFile.write(str(execTimes)+'\n')
    end_time = time.time()
    print 'total execution time is %f seconds' % (end_time - sstart_time)
    print 'distEncode_time = %f seconds' % distEncode_time
    print sum_Lx
    
    if last:
        cmd = "bash /home/krishna/finalApps/killSlavesDistributed.sh"
        os.system(cmd)
    resultFile.close() 

def main():
    np.random.seed(1351)
    total = 3
    resultFileName = sys.argv[1]
    print resultFileName 
    for trial in range(total):
        if not trial:
            run(first=True, resultFileName=resultFileName)
        elif trial == total - 1:
            run(last=True, resultFileName=resultFileName)
        else:
            run(resultFileName=resultFileName)

if __name__ == '__main__':
    main()
