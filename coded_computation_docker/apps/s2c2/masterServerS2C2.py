from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
import xmlrpclib
import socket
import fcntl
import struct
import multiprocessing
import numpy as np
import sys
import time
from assignImprovedGeneric import assignRnLImproved
import json
import re
import os
import signal
from collections import defaultdict
import shared_vars
import lstmForSpeed_container as lstm

#configs = json.load(open('/home/zhifeng/apps/s2c2/config/config.json'))
configs = json.load(open('./config/config.json'))
encoding = np.array(configs['matrixConfigs']['encoding'])
k, n = encoding.shape

READY_SLAVES_NEEDED = n
CODING_COPIES_NEEDED = n #Changed for this case 
CHUNKS = configs['chunks'] 
#CHUNKS = 1
def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Register an instance; all the methods of the instance are
# published as XML-RPC methods (in this case, just 'div').
class MyFuncs:
    def __init__(self, token, fast, threshold, ready, finishTimes):
        self.fast = fast
        self.token = token
        self.ready = ready
        self.threshold = threshold
        self.products = {}
        self.slavePids = {}
        self.readyCount = 0
        self.finishTimes = finishTimes
    
    def slave_ready(self, slaveID):
        print("slave_ready")
        self.readyCount += 1
        print("slave %s is ready" % (slaveID))
        if self.readyCount >= READY_SLAVES_NEEDED:
            self.ready.clear()
            self.ready.set()
        return
    
    def checkDone(self):
        return len(self.products.keys()) >= CODING_COPIES_NEEDED
       
    def accept_pid(self, slave, pid):
        self.slavePids[slave] = pid

    def accept_product(self, product, partition):
        mat = re.match(r"slave(\d+)",partition)
        ID = (int)(mat.group(1))
        self.finishTimes[ID-1] = time.time()
	if len(self.products.keys()) < CODING_COPIES_NEEDED:
            self.products[partition] = product
        if len(self.products.keys()) >= self.threshold:
            self.fast.clear()
            self.fast.set()
        if len(self.products.keys()) >= CODING_COPIES_NEEDED:
            self.token.clear()
            self.token.set()
        return

    def clear(self):
        self.products = {}
        self.slavePids = {}
        self.token.clear()
        self.fast.clear()
        self.ready.clear()
        self.finishTimes = np.zeros(READY_SLAVES_NEEDED)

    def retrieve_products(self):
        return self.products

    def retrieve_pids(self):
        return self.slavePids

    def retrieve_finishTimes(self):
	return self.finishTimes.tolist()

class MasterServerProcess(multiprocessing.Process):
    def __init__(self, myIP, myPortNum, token, fast, threshold, ready, finishTimes):
        multiprocessing.Process.__init__(self)
        #self.setDaemon(True)
        self.daemon = True
        self.server = SimpleXMLRPCServer((myIP, int(myPortNum)),
                                     requestHandler=RequestHandler, allow_none=True)
        self.server.register_introspection_functions()
        
        myFuncs = MyFuncs(token, fast, threshold, ready, finishTimes)
        self.funcs = myFuncs
        self.server.register_instance(myFuncs)

    def run(self):
        self.server.serve_forever()

def encode_matrix(matrix, encoding):
    k, n = encoding.shape
    splits = np.array_split(matrix, k)
    encodeds = []
    for idx in range(n):
      code = encoding[:, idx]
      encoded = np.zeros_like(splits[0])
      for split_idx, coeff in enumerate(code):
          encoded += coeff * splits[split_idx]
      ptFile = '/home/zhifeng/apps/s2c2/data/partition%d.mat' % (idx+1)
      np.savetxt(ptFile, encoded, fmt='%.5f')
      cmd = ("scp %s slave%d:%s" % (ptFile, idx+1, ptFile)) 
      os.system(cmd)

      encodeds.append(encoded)
    
    return encodeds   
   
def encode_matrix_tp(matrix, encoding):
    k, n = encoding.shape
    splits = np.array_split(matrix, k)
    encodeds = []
    for idx in range(n):
      code = encoding[:, idx]
      encoded = np.zeros_like(splits[0])
      for split_idx, coeff in enumerate(code):
          encoded += coeff * splits[split_idx]
      ptFile = '/home/zhifeng/apps/s2c2/data/partition%d_tp.mat' % (idx+1)
      np.savetxt(ptFile, encoded, fmt='%.5f')
      cmd = ("scp %s slave%d:%s" % (ptFile, idx+1, ptFile)) 
      os.system(cmd)

      encodeds.append(encoded)
    
    return encodeds   
         
def decode_products(products, dim, k, n):
    #result = np.zeros((dim, dim))
    result = np.zeros((dim, 1))
    p = re.compile('slave(\d+)')
    last_product = None
    s = set(np.arange(k) + 1)
    for slave in sorted(products.keys(), key=lambda k: int(k[len('slave'):])):
        product = products[slave]
        ID = int(p.match(slave).group(1))
        if ID != n:
            start = (ID - 1) * (dim/k)
            end = start + dim/k
            result[start:end, :] = product
            s.remove(ID)
        else:
            last_product = product
    if s:
        missing = s.pop()
        start = (missing - 1) * (dim/k)
        end = start + dim/k
        result_list = np.array(np.split(result, k))
        result[start:end,:] = last_product - result_list.sum(axis=0)
    return result
 
def decode_products_generic(encoding, lookup, products, dim, k, n, decodeInfo):
    #result = np.zeros((dim, dim))
    result = np.zeros((dim, 1))
    p = re.compile('slave(\d+)')
    s = set(np.arange(k) + 1)
    cols = []
    slaveDict = {}
    for slave in sorted(products.keys(), key=lambda k: int(k[len('slave'):])):
        product = products[slave]
        ID = int(p.match(slave).group(1))
        cols.append(ID-1)
        slaveDict[ID-1] = product
        if ID <= k:
            start = (ID - 1) * (dim/k)
            end = start + dim/k
            if product.shape[0] != (dim/k):
                print slave + ': race condition! Ignore for now.'
            else:
                result[start:end, :] = product
            s.remove(ID)
    for missing in s:
        start = (missing - 1) * (dim/k)
        end = start + dim/k
        result_list = np.array(np.split(result, k))
        encodedM = encoding[:, cols]
        #print retIdx, value, cols
        # k results are available. k x k matrix
        # k (rows correspond to splits of original matrix A) x k(columns correspond to results available) matrix
        # On inversion to k (available results) x k (splits of the matrix A)
        # It is possible to obtain split A[x] by indexing the coefficients of corresponding column and taking linear combination of appropriate results with coefficients
        encodedMstr = str(encodedM)
        if encodedMstr not in lookup.keys():
            lookup[encodedMstr] = np.linalg.inv(encodedM)
        encodedM_inv = lookup[encodedMstr]
        coeffs = encodedM_inv[:, missing-1]
        tmp = None
        for index in range(len(cols)):
            try:
                sId = cols[index]
                if not index:
                    tmp = coeffs[index] * slaveDict[sId]
                else:
                    #print sId, value - rows[sId]
                    tmp += coeffs[index] * slaveDict[sId]
            except:
                print 'race condition! Ignore for now.'
                
        result[start:end,:] = tmp 
    return result
 
def decode_products_generic_improved(encoding, lookupDict, products, dim, k, n, decodeInfoDict, chunkAssignmentDict, rows_slaveDict, lengthsDict):
    result = np.zeros((dim, 1)) #Vector
    # to bypass race condition related exceptions while decoding, and to collect performance numbers, return.
    return result
    p = re.compile('slave(\d+)')
    s = set(np.arange(k) + 1)
    cols = []
    colsReceived = []
    slaveDict = {}
    for slave in sorted(products.keys(), key=lambda k: int(k[len('slave'):])):
        #Store products received into a dictionary
        product = products[slave]
        ID = int(p.match(slave).group(1))
        colsReceived.append(ID-1) #Received products
        slaveDict[ID-1] = product
    #print "decode:"
    for chunk, assigned in decodeInfoDict.iteritems():
        #Decode and construct missing products
        s = set(np.arange(k)) #first k products
        cols = list(assigned)
        if not lengthsDict[cols[0]][chunk]:
            continue
        start = rows_slaveDict[cols[0]][chunk] #chunk * (length of number of rows in chunk)
        end = start+lengthsDict[cols[0]][chunk]-1
        #print "chunk is:", chunk
        #print start, end
        filt = filter(lambda x: x<k, cols)
        for i in filt:
            s.remove(i) #Remove any of the first k products that are received. Remaining among k need to be constructed
        encodedM = encoding[:, cols]
        encodedMstr = str(encodedM)
        if encodedMstr not in lookupDict.keys():
            try:
                lookupDict[encodedMstr] = np.linalg.inv(encodedM)
            except:
                print 'corner case!! Ignore for now.'
                continue
        encodedM_inv = lookupDict[encodedMstr]
        tmpResult = None
        for missing in s:
            coeffs = encodedM_inv[:, missing]
            for index, slaveID in enumerate(cols):#Decode using the products received from slaves
            #Note: The products returned from each slave may be vertical stack of disjoint product chunks computed. Indexes to find the correct range of product to use for decode
                startIndex = endIndex = 0
                for i in xrange(chunk+1):
                    if i<chunk:
                        startIndex += chunkAssignmentDict[slaveID][i] * lengthsDict[slaveID][i]
                    endIndex += chunkAssignmentDict[slaveID][i] * lengthsDict[slaveID][i]
                if not index:
                    tmpResult =  coeffs[index] * slaveDict[slaveID][startIndex:(endIndex-1)]
                else:
                    try:
                        tmpResult +=  coeffs[index] * slaveDict[slaveID][startIndex:(endIndex-1)]
                    except:
                        #print 'tmpResult shape is %s, current slaveID is %d, startIndex is %d and endIndex is %d' % (str(tmpResult.shape), slaveID, startIndex, endIndex)
                        #print 'chunkAssignmentDict for it is %s, lengthsDict for it is %s' % (str(chunkAssignmentDict[slaveID]), str(lengthsDict[slaveID]))
                        print 'EXCEPTION in decode generic: start is %d, end is %d, final result shape is %s' % (start, end, str(result.shape))
        result[start:end,:] = tmpResult
    #result[start:end,:] = tmpResult
    return result

def generateReplicasAndSpeeds(means):
    #replicas = map(lambda m: int(np.round(1.0+np.random.exponential(m-1.0+0.1))), means)
    means = [1] * READY_SLAVES_NEEDED
    replicas = means
    totalReplicas = sum(replicas) 
    speeds = (1.0) / np.array(replicas)
    speeds = map(lambda x: int(x), speeds)
    replicas = map(lambda r: r*100, replicas)
    return replicas, speeds
             
def generateReplicasAndSpeedsVarying(means):
    means = map(lambda x: int(x), means)
    dictRM = {10:5,12:4}
    dictRM = defaultdict(lambda: 0, dictRM)
    replicaList = [np.random.choice(dictRM.keys()) for i in means]
    replicas = map(lambda pair: int(pair[0]*pair[1]), zip(replicaList, means)) 
    speeds = map(lambda x: dictRM[x], replicas)
    return replicas, speeds

def loadHistory():
    for ID in range(n):
        shared_vars.histories[ID] = np.ones(shared_vars.histLen)

def predictSpeedsLSTM():
    ui = np.ones(n)
    for i in range(n):
        ui[i] = shared_vars.histories[i][-1]
    ui = lstm.predict_next(ui, shared_vars.model)
    return ui

def main():
    matrixMulKernelMaster()

def matrixMulKernelMaster(iteration=0, matrix=None, execTimes=None, isTranspose=False):
    #np.random.seed(1351)
    configs = json.load(open('/home/zhifeng/apps/s2c2/config/config.json'))
    #myIP = configs['masterConfigs']['IP']
    myIP = get_ip_address('eth0')
    myPortNum = configs['masterConfigs']['PortNum']

    start_time = time.time()
    global token, threshold, fast, ready, server_process, encoding, encodeds, slaves, localProxy, dim, finishTimes

    # Create server
    if not iteration:
        loadHistory()
        encoding = np.array(configs['matrixConfigs']['encoding'])
        k, n = encoding.shape
        threshold = n-3
        token = multiprocessing.Event()
        fast = multiprocessing.Event()
        ready = multiprocessing.Event()
        finishTimes = np.zeros(n)
        server_process = MasterServerProcess(myIP, myPortNum, token, fast, threshold, ready, finishTimes) 
        server_process.start()
        print 'starting master server process...'
        slaves = []
        for i in range(1,n+1):
            slave = configs['slaveConfigs']['slave'+str(i)]['IP'] + ':' + configs['slaveConfigs']['slave'+str(i)]['PortNum']
            slaves.append(xmlrpclib.ServerProxy('http://' + slave, allow_none=True))

        localProxy = xmlrpclib.ServerProxy('http://' + myIP + ':' + myPortNum, allow_none=True)
        
   
        if matrix is None: 
            dim = configs['matrixConfigs']['dimension']
            matrix = np.random.rand(dim,dim)
        else:
            dim = matrix.shape[0]
        #encodeds = encode_matrix(matrix, encoding)
        
        #execTimes = configs['execTimes']

        end_time = time.time()
        #print 'finish distributing with %f seconds' % (end_time - start_time)
        start_time = end_time


    #start_time = time.time()
    dim = matrix.shape[0]
    localProxy.clear()
    k, n = encoding.shape
    chunks = CHUNKS
    decodeInfo = {}
    chunks_replicas = {i : [] for i in range(n)}
    chunks_rows = {i : [] for i in range(n)}
    chunks_lengths = {i : [] for i in range(n)}
    rows = [0 for i in range(n)]
    lengthsDef = [((dim/k)/chunks) for i in range(n)]
    if not iteration:
        print("wait for all slaves to get ready")
        ready.wait()
    #load_time = time.time() - start_time
    start_time = time.time()
    startTimes = np.zeros(n)
    finishTimes= np.zeros(n)
    for chunk in range(chunks):#Number of chunks = 1 
        #ui = getSlaveSpeeds()
        replicas, ui = generateReplicasAndSpeeds(execTimes)
        predict_start =  time.time()
        ui = predictSpeedsLSTM()
        print("predicted speed ratios is:", ui)
        print("Prediction Time: ", time.time() - predict_start)
        #replicas, ui = generateReplicasAndSpeedsVarying(execTimes)
        print replicas
        rows_slaveDict, lengthsDict, decodeInfoDict, chunkAssignmentDict, countMap = assignRnLImproved(ui, n, k, dim/k)
        for i in range(n):
            #print len(rows_slaveDict[0])
            chunks_replicas[i] = [replicas[i]]*len(rows_slaveDict[0])
            #print 'replicas', str(chunks_replicas[i])
            chunks_rows[i] = rows_slaveDict[i]
            chunks_lengths[i] = lengthsDict[i]
            print 'chunk lengths for slave %d is %s' % (i, str(chunks_lengths[i]))
            shared_vars.slaveLengths[i] = np.sum(lengthsDict[i])
    communication_time = 0
    c_time = time.time()
    # distribute the data
    for i in range(n):
      print("slave id is:%d" % i)
      slaves[i].accept_matrix(chunks_rows[i], chunks_lengths[i], chunks_replicas[i])
    #for i in range(n): 
    #    slaves[i].start()
      startTimes[i] = time.time()
    communication_time = time.time() - c_time
    ###BEGIN - Coded added for robustness###
    print("waiting for fast workers to complete")
    fast.wait()
    timeTaken = np.zeros(threshold)
    finishTimes = np.array(localProxy.retrieve_finishTimes())
    i = 0 
    for idx in xrange(n):
        if finishTimes[idx] > 0 and i < threshold:
            timeTaken[i] = finishTimes[idx] - start_time
            print 'time taken for slave %d is: %f' % (idx+1, timeTaken[i])
            i += 1

    minIndex = np.argsort(timeTaken)
    average = np.mean(timeTaken)
    print("timeTaken, average time taken: ", timeTaken, average)
    #wait for stdev seconds
    print("waiting for stdevs time, minimum index: ", minIndex)
    token.wait(0.15*average)
    finishTimes = np.array(localProxy.retrieve_finishTimes())
    slowSlaves = np.where(finishTimes == 0)[0] + 1
    if len(slowSlaves):
        print 'mis_predict'
    else:
        print 'correctly_predict'

    for i in slowSlaves:
        print 'releasing slave %d' % (i-1)
        c_time = time.time()
        slaves[i-1].release()
        communication_time += time.time() - c_time
    #token.wait()
    ###END - Code added for robustness###
    productFileNames = localProxy.retrieve_products()
    finishTimes = np.array(localProxy.retrieve_finishTimes())
    shared_vars.slaveExecTimes = finishTimes-startTimes
    for ID in range(n):
        shared_vars.histories[ID] = np.delete(shared_vars.histories[ID],0)
        actualSpeed = shared_vars.slaveLengths[ID]
        #print actualSpeed
        if not isTranspose:
            actualSpeed = actualSpeed * (1.0) * (1.0/shared_vars.slaveExecTimes[ID])
        else:
            # account for the fact that we are doing transpose multplication!!
            print 'doing transpose!!! dim = %d' % dim
            actualSpeed = actualSpeed * (14.0) * (1.0/shared_vars.slaveExecTimes[ID])
        print "slave: ",ID, " measured speed is: ",actualSpeed
        shared_vars.histories[ID] = np.append(shared_vars.histories[ID], actualSpeed)

	end_time = time.time()
    computeTime = end_time - start_time - communication_time
    print 'I got what I need from slaves with %f seconds' % (end_time - start_time)
    start_time = end_time

    products = {}
    p = re.compile('slave(\d+)')
    #s = set(np.arange(n) + 1)
    for slave in sorted(productFileNames.keys()):
        ID = int(p.match(slave).group(1))
   
    lookup = {}
    #result = decode_products(products, dim, k, n)
    result = decode_products_generic_improved(encoding, lookup, products, dim, k, n, decodeInfoDict, chunkAssignmentDict, rows_slaveDict, lengthsDict)
    end_time = time.time()
    decode_time = end_time - start_time
    #print 'finish loading and decoding with %f seconds' % (end_time - start_time)
    
    localProxy.clear()
    #verify = matrix.dot(np.arange(dim*dim).reshape(dim,dim) + 1)
    #print 'verifying results, maximum of the element difference is', np.max(np.abs(result-verify))
    return result, computeTime, communication_time, decode_time

if __name__ == '__main__':
    main()
