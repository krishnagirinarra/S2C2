from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
import xmlrpclib
import socket
import fcntl
import struct
import multiprocessing
import numpy as np
import sys
import json
import time
import os

configs = json.load(open('/home/zhifeng/apps/s2c2/config/config.json'))
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
    def __init__(self, token, mvToken):
        self.token = token
        self.mvToken = mvToken
        self.rows = None
        self.lengths = None
        self.replicas = None
        self.isReleased = False

    def accept_matrix(self, rows, lengths, replicas=1):
        self.rows = rows
        self.lengths = lengths
        self.replicas = replicas
        self.start()
        return
    
    def start(self):
        self.token.clear()
        self.token.set()
        return

    def retrieve_matrix(self):
        return self.rows, self.lengths, self.replicas

    def release(self):
        self.isReleased = True
        self.mvToken.clear()
        self.mvToken.set()
        return

    def is_released(self):
        ret = self.isReleased
        self.isReleased = False
        return ret

class SlaveServerProcess(multiprocessing.Process):
    def __init__(self, myIP, myPortNum, token, mvToken):
        multiprocessing.Process.__init__(self)
        #self.setDaemon(True)
        self.daemon = True
        self.server = SimpleXMLRPCServer((myIP, int(myPortNum)),
                                     requestHandler=RequestHandler, allow_none=True)
        self.server.register_introspection_functions()
        
        myFuncs = MyFuncs(token, mvToken)
        self.funcs = myFuncs
        self.server.register_instance(myFuncs)

    def run(self):
        self.server.serve_forever()

def main():
    if len(sys.argv) < 3:
        print 'incorrect number of arguments'
        print 'please provide id of the slave server, N/T to indicate transpose or not' 
        sys.exit(-1)

    ID = sys.argv[1]
    TP = sys.argv[2]
    matrixMulKernelSlave(ID, TP)

def matrixMultiply(matrix, vector_o, matrixTP, rows, lengths, replicas, mvToken, fullLength):
    chunks = CHUNKS
    product = None
    rowList = []
    for i in xrange(len(rows)): #number of chunks is 1
        row = rows[i]
        length = lengths[i]
        rowList += range(row,row+length)
    matrixConst = matrix[rowList,:]
    print("shape of replica matrix: ", matrixConst.shape)
    replicasTracking = replicas[0]
    for chunk in range(chunks):
        for i in range(1, 31):
            replicasTracking -= 1
            wasteFile = open('wasteSlave.out', 'a')
            wasteFile.write(str(replicasTracking) + '\n')
            wasteFile.close()
            timeB = time.time()
            vector = np.diag(vector_o).dot(matrixTP)
            product = np.zeros((matrixConst.shape[0], matrixTP.shape[1]))
            for rIdx in range(int(matrixConst.shape[0])):
                #print("shapes are: ", vector.shape, )
                product[rIdx] = matrixConst[rIdx].dot(vector)
            #product = matrixConst.dot(vector)
            timeE = time.time()
            print("replica computing takes %s seconds:" % (timeE-timeB))
        print("replicas are: ", replicas[0])
    mvToken.clear()
    mvToken.set()

def matrixMulKernelSlave(ID, TP, vector=None, matrix=None):
    configs = json.load(open('/home/zhifeng/apps/s2c2/config/config.json'))
    master = configs['masterConfigs']['IP'] + ':' + configs['masterConfigs']['PortNum']
    #myIP = configs['slaveConfigs']['slave' + ID]['IP']
    myIP = get_ip_address('eth0')
    myPortNum = configs['slaveConfigs']['slave' + ID]['PortNum']

    # Create server
    token = multiprocessing.Event()
    mvToken = multiprocessing.Event()

    if matrix is None:
        start_time = time.time()
        matrix = np.loadtxt('/home/zhifeng/apps/s2c2/data/poly_partition_AT%s.mat' % ID, dtype=int)
        matrixTP = np.loadtxt('/home/zhifeng/apps/s2c2/data/poly_partition_A%s.mat' % ID, dtype=int)
        load_time = time.time() - start_time
        f = open('/home/zhifeng/apps/s2c2/data/slaveLoadTime%s.out' % ID, 'w')
        f.write(str(load_time)+'\n')
        f.close()

    #matrices = (matrix, matrixTP)
    matrices = (matrix, matrix)
    server_process = SlaveServerProcess(myIP, myPortNum, token, mvToken) 

    server_process.start()
    print 'starting slave server process %d...' % server_process.pid

    localProxy = xmlrpclib.ServerProxy('http://' + myIP + ':' + myPortNum, allow_none=True)
    masterProxy = xmlrpclib.ServerProxy('http://' + master, allow_none=True)
    chunks = CHUNKS
    idx = 0
    while True:
        try:
            masterProxy.slave_ready(ID)
            break
        except:
            print("master did not start/accept ACK.")
            time.sleep(1)
            pass

    index = 0 
    print 'entering computation'
    while True:
        index += 1
        matrix = matrices[idx%2]
        idx += 1
        product = None
        for chunk in range(chunks):##Assumption: 1 chunk
            if not chunk:
                token.wait()
                token.clear()
                mvToken.clear()
                localProxy.is_released()
                timeB = time.time()
                vector_o = np.random.rand(matrix.shape[1]) * 1
                print("shape of vector",vector_o.shape)
                print("shape of matrix",matrix.shape)
                vector = np.diag(vector_o).dot(matrixTP)
                timeE = time.time()
                print 'computing Diagonal product takes %s seconds' % str(timeE - timeB)

                print("retrieving matrix...")
                rows, lengths, replicas = localProxy.retrieve_matrix()
                print 'slave' + ID +': get my share of data and start to compute'
                start_time = time.time()
            rowList=[]
            for i in xrange(len(rows)): #number of chunks is 1
                row = rows[i]
                length = lengths[i]
                rowList += range(row,row+length)
            
            timeB = time.time()
            #print("rowlist is: ", rowList)
            matrixConst = matrix[rowList,:]
            timeE = time.time()
            print 'gathering rows takes %s seconds' % str(timeE - timeB)
            timeB = time.time()
            product = np.zeros((matrixConst.shape[0], matrixTP.shape[1]))
            print("length is: ", matrixConst.shape[0])
            print("shape of vector",vector.shape)
            print("shape of matrixConst[rIdx]",matrixConst[0].shape)
            for rIdx in range(int(matrixConst.shape[0])):
                product[rIdx] = matrixConst[rIdx].dot(vector)
            #product = matrixConst.dot(vector)
            timeE = time.time()
            print 'computing takes %s seconds' % str(timeE - timeB)

        #mv = multiprocessing.Process(target=matrixMultiply, args=(matrixConst, vector, rows, lengths, replicas, mvToken, matrix.shape[0]))
        mv = multiprocessing.Process(target=matrixMultiply, args=(matrix, vector_o, matrixTP, rows, lengths, replicas, mvToken, matrix.shape[0]))
        mv.start()
        mvToken.wait()
        mvToken.clear()
        mv.terminate()             
        end_time = time.time()
        compTime = end_time - start_time
        print 'slave' + ID + ': time to compute: %f' % (end_time - start_time)
        start_time = end_time
        #if not masterProxy.checkDone():
        #if not localProxy.is_released():
        if True:
            resultFile = open('resultSlave%s.out' % ID, 'a')
            resultFile.write(str(index)+', ')
            resultFile.write(str(compTime)+'\n')
            resultFile.close()
            productFile = ''
            if TP == 'N':
                productFile = '/home/zhifeng/apps/s2c2/data/product%s.mat' % ID
            else:
                productFile = '/home/zhifeng/apps/s2c2/data/product%s_tp.mat' % ID
            #print("writing to product file")
            #print product
            np.savetxt(productFile, product)
            #cmd = 'scp %s master:%s' % (productFile, productFile) 
            masterIP = configs['masterConfigs']['IP']
            cmd = "scp -P 5000 %s %s:%s" % (productFile, masterIP, productFile) 
            os.system(cmd)
            end_time = time.time()
            print 'slave' + ID + ': time to "send" result: %f' % (end_time - start_time)
            timeAB = time.time()
            masterProxy.accept_product(productFile, 'slave' + ID)
            timeAE = time.time()
            print 'Accept sent to master takes time: %f' % (timeAE-timeAB)
        else:
            print 'slave'+ ID + ': I am too slow and the master has what it needs'
        
    server_process.terminate()

if __name__ == '__main__':
    main()   
