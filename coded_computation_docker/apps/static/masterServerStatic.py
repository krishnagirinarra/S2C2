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
from assign import assignRnL
import json
import re
import os
import signal

configs = json.load(open('/home/zhifeng/apps/static/config/config.json'))
encoding = np.array(configs['matrixConfigs']['encoding'])
k, n = encoding.shape

READY_SLAVES_NEEDED = n
CODING_COPIES_NEEDED = k 
CHUNKS = configs['chunks'] 

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
    def __init__(self, token, ready):
        self.token = token
        self.ready = ready
        self.readyCount = 0
        self.products = {}
        self.slavePids = {}
 
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
        if len(self.products.keys()) < CODING_COPIES_NEEDED:
            self.products[partition] = product
        if len(self.products.keys()) >= CODING_COPIES_NEEDED:
            self.token.clear()
            self.token.set()
        return

    def clear(self):
        self.products = {}
        self.slavePids = {}
        self.token.clear()
        self.ready.clear()

    def retrieve_products(self):
        return self.products

    def retrieve_pids(self):
        return self.slavePids

class MasterServerProcess(multiprocessing.Process):
    def __init__(self, myIP, myPortNum, token, ready):
        multiprocessing.Process.__init__(self)
        #self.setDaemon(True)
        self.daemon = True
        self.server = SimpleXMLRPCServer((myIP, int(myPortNum)),
                                     requestHandler=RequestHandler, allow_none=True)
        self.server.register_introspection_functions()
        
        myFuncs = MyFuncs(token, ready)
        self.funcs = myFuncs
        self.server.register_instance(myFuncs)

    def run(self):
        self.server.serve_forever()

def encode_matrix(matrix, encoding):
    k, n = encoding.shape
    splits = np.array_split(matrix, k)
    encodeds = []
    for idx in range(n):
      if idx < k:
          encoded = splits[idx]
      else:
          code = encoding[:, idx]
          encoded = np.zeros_like(splits[0])
          for split_idx, coeff in enumerate(code):
              encoded += coeff * splits[split_idx]
      ptFile = '/home/zhifeng/apps/static/data/more/partition%d.mat' % (idx+1)
      np.savetxt(ptFile, encoded)
      cmd = ("scp %s slave%d:%s" % (ptFile, idx+1, ptFile)) 
      os.system(cmd)

      encodeds.append(encoded)
    
    return encodeds   
   
def encode_matrix_tp(matrix, encoding):
    k, n = encoding.shape
    splits = np.array_split(matrix, k)
    encodeds = []
    for idx in range(n):
      if idx < k:
          encoded = splits[idx]
      else:
          code = encoding[:, idx]
          encoded = np.zeros_like(splits[0])
          for split_idx, coeff in enumerate(code):
              encoded += coeff * splits[split_idx]
      ptFile = '/home/zhifeng/apps/static/data/more/partition%d_tp.mat' % (idx+1)
      np.savetxt(ptFile, encoded)
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
 
def decode_products_generic(encoding, lookup, products, dim, k, n):
    #result = np.zeros((dim, dim))
    result = np.zeros((dim, 1))
    # to bypass race condition related exceptions while decoding, and to collect performance numbers, return.
    return result
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
       
def generateReplicasAndSpeeds(means):
    replicas = map(lambda m: int(np.round(1.0+np.random.exponential(m-1.0+0.1))), means)
    replicas = means
    speeds = 1.0 / np.array(replicas)
    return replicas, speeds
             
def main():
    matrixMulKernelMaster()

def matrixMulKernelMaster(iteration=0, matrix=None, execTimes=None):
    #np.random.seed(1351)
    configs = json.load(open('/home/zhifeng/apps/static/config/config.json'))
    #myIP = configs['masterConfigs']['IP']
    myIP = get_ip_address('eth0')
    myPortNum = configs['masterConfigs']['PortNum']

    start_time = time.time()
    global token, ready, server_process, encoding, encodeds, slaves, localProxy, dim

    # Create server
    if not iteration:
        token = multiprocessing.Event()
        ready = multiprocessing.Event()
        server_process = MasterServerProcess(myIP, myPortNum, token, ready) 
        server_process.start()
        print 'starting master server process...'
        slaves = []
        encoding = np.array(configs['matrixConfigs']['encoding'])
        k, n = encoding.shape
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


    dim = matrix.shape[0]
    localProxy.clear()
    k, n = encoding.shape
    chunks = CHUNKS
    chunks_replicas = {i : [] for i in range(n)}
    chunks_rows = {i : [] for i in range(n)}
    chunks_lengths = {i : [] for i in range(n)}
    rows = [0 for i in range(n)]
    lengths = [((dim/k)/chunks) for i in range(n)]
    if not iteration:
        print("wait for all slaves to get ready")
        ready.wait()
    start_time = time.time()

    for chunk in range(chunks): 
        #ui = getSlaveSpeeds()
        replicas, ui = generateReplicasAndSpeeds(execTimes)
        print replicas
        rows_slave = np.array(rows)
        rows_slave += chunk * (dim/k)/chunks
        print rows_slave, lengths
        for i in range(n):
            chunks_replicas[i].append(replicas[i])
            chunks_rows[i].append(int(rows_slave[i]))
            chunks_lengths[i].append(int(lengths[i]))

    communication_time = 0
    c_time = time.time()
    # distribute the data
    for i in range(n):
      slaves[i].accept_matrix(chunks_rows[i], chunks_lengths[i], chunks_replicas[i])
    #for i in range(n): 
    #    slaves[i].start()
    communication_time += time.time() - c_time

    token.wait()
    productFileNames = localProxy.retrieve_products()
    end_time = time.time()
    computeTime = end_time - start_time - communication_time
    #print 'I got what I need from slaves with %f seconds' % (end_time - start_time)
    start_time = end_time

    products = {}
    p = re.compile('slave(\d+)')
    s = set(np.arange(n) + 1)
    for slave in sorted(productFileNames.keys()):
        ID = int(p.match(slave).group(1))
        s.remove(ID)
    for i in s:
        print 'releasing %d' % i
        c_time = time.time()
        slaves[i-1].release()
        communication_time += time.time() - c_time
    
    lookup = {}
    #result = decode_products(products, dim, k, n)
    result = decode_products_generic(encoding, lookup, products, dim, k, n)
    end_time = time.time()
    decode_time = end_time - start_time
    #print 'finish loading and decoding with %f seconds' % (end_time - start_time)
    
    localProxy.clear()
    #verify = matrix.dot(np.arange(dim*dim).reshape(dim,dim) + 1)
    #print 'verifying results, maximum of the element difference is', np.max(np.abs(result-verify))
    return result, computeTime, communication_time, decode_time

if __name__ == '__main__':
    main()
