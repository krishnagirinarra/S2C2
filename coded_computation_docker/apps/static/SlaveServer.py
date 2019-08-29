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
#import mkl
import os

#mkl.set_num_threads(1)
configs = json.load(open('/home/zhifeng/apps/static/config/config.json'))
CHUNKS = configs['chunks'] 
replicasTracking = [100]

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

def matrixMultiply(matrix, vector, rows, lengths, replicas, mvToken, timeTotal):
    chunks = CHUNKS
    product = None
    print 'I am doing extra work'
    print 'one replica takes %s seconds' % str(timeTotal)
    for chunk in range(chunks):
        row = rows[chunk]
        length = lengths[chunk]
        replicasTracking[0] = replicas[chunk]
        for i in range(1, replicas[chunk]):
#             time.sleep(timeTotal[chunk])
            replicasTracking[0] -= 1
            wasteFile = open('wasteSlave.out', 'a')
            wasteFile.write(str(replicasTracking[0]) + '\n')
            wasteFile.close()

            if not chunk:
                product = np.zeros((length, 1)) 
                for rIdx in range(length):
                    try:
                        product[rIdx] = matrix[(row+rIdx), :].dot(vector)
                    except:
                        print 'race condition!!!'
                        break
                #product = matrix[row : (row+length), :].dot(vector)
            elif not i:
                product = np.vstack((product, matrix[row : (row+length), :].dot(vector)))
            else:
                matrix[row : (row+length), :].dot(vector)
        print replicas[chunk]
    mvToken.clear()
    mvToken.set()

def matrixMulKernelSlave(ID, TP, vector=None, matrix=None):
    configs = json.load(open('/home/zhifeng/apps/static/config/config.json'))
    master = configs['masterConfigs']['IP'] + ':' + configs['masterConfigs']['PortNum']
    #myIP = configs['slaveConfigs']['slave' + ID]['IP']
    myIP = get_ip_address('eth0')
    myPortNum = configs['slaveConfigs']['slave' + ID]['PortNum']
    execTimes = configs['execTimes']

    # Create server
    token = multiprocessing.Event()
    mvToken = multiprocessing.Event()

    if matrix is None:
        start_time = time.time()
        matrix = np.loadtxt('/home/zhifeng/apps/static/data/partition%s.mat' % ID, dtype=int)
        matrixTP = np.loadtxt('/home/zhifeng/apps/static/data/partition%s_tp.mat' % ID, dtype=int)
        load_time = time.time() - start_time
        f = open('/home/zhifeng/apps/static/data/slaveLoadTime%s.out' % ID, 'w')
        f.write(str(load_time)+'\n')
        f.close()

    matrices = (matrix, matrixTP)
    server_process = SlaveServerProcess(myIP, myPortNum, token, mvToken) 

    server_process.start()
    print 'starting slave server process %d...' % server_process.pid

    localProxy = xmlrpclib.ServerProxy('http://' + myIP + ':' + myPortNum, allow_none=True)
    masterProxy = xmlrpclib.ServerProxy('http://' + master, allow_none=True)
    chunks = CHUNKS
    idx = 0
    ownReplicas = np.array([100]*1000) 
    print ownReplicas.min(), ownReplicas.max()
    np.savetxt('/home/zhifeng/apps/static/data/replicas%s.out' % ID, ownReplicas.reshape(-1,1), fmt='%f') 
    rIndex = 0
    while True:
        try:
            masterProxy.slave_ready(ID)
            break
        except:
            print("master did not start/accept ACK.")
            time.sleep(1)
            pass

    index = 0
    while True:
        index += 1
        matrix = matrices[idx % 2]
        idx += 1
        product = None
        timeTotal = []
        for chunk in range(chunks):
            if not chunk:
                token.wait()
                token.clear()
                mvToken.clear()
                localProxy.is_released()
                vector = None

                while vector is None or vector.shape[0] != matrix.shape[1]:
                    try: 
                        vector = np.random.rand(matrix.shape[1], 1)
                    except:
                        print 'race condition due to NFS'

                rows, lengths, replicas = localProxy.retrieve_matrix()
                for j in range(len(replicas)):
                    replicas[j] = int(ownReplicas[rIndex])
                    rIndex += 1
                    if rIndex >= ownReplicas.shape[0]:
                        rIndex = 0 
                #print 'slave' + ID +': get my share of data and start to compute'
                start_time = time.time()
            row = rows[chunk]
            length = lengths[chunk]
            timeB = time.time()
            if not chunk:
                product = np.zeros((length, 1)) 
                for rIdx in range(length):
                    try: 
                        product[rIdx] = matrix[(row+rIdx), :].dot(vector)
                    except:
                        print 'super slow causing race condition!!'
                        localProxy.release()
                        break
                #product = matrix[row : (row+length), :].dot(vector)
            else:
                product = np.vstack((product, matrix[row : (row+length), :].dot(vector)))
            timeE = time.time()
            timeTotal.append(timeE - timeB)
        if not localProxy.is_released():
            mv = multiprocessing.Process(target=matrixMultiply, args=(matrix, vector, rows, lengths, replicas, mvToken, timeTotal))
            mv.start()
            mvToken.wait()
            mvToken.clear()
            mv.terminate()             
        end_time = time.time()
        print 'slave' + ID + ': time to compute: %f' % (end_time - start_time)
        compTime = end_time - start_time 
        start_time = end_time
        #if not masterProxy.checkDone():
        if not localProxy.is_released():
            resultFile = open('resultSlave%s.out' % ID, 'a')
            resultFile.write(str(index)+', ')
            resultFile.write(str(compTime)+'\n')
            resultFile.close()
            productFile = ''
            if TP == 'N':
                productFile = '/home/zhifeng/apps/static/data/product%s.mat' % ID
            else:
                productFile = '/home/zhifeng/apps/static/data/product%s_tp.mat' % ID

            np.savetxt(productFile, product)
            #cmd = 'scp %s master:%s' % (productFile, productFile)
            masterIP = configs['masterConfigs']['IP']
            cmd = "scp -P 5000 %s %s:%s" % (productFile, masterIP, productFile) 
            os.system(cmd)
            end_time = time.time()
            #print 'slave' + ID + ': time to "send" result: %f' % (end_time - start_time)
            masterProxy.accept_product(productFile, 'slave' + ID)
        else:
            print 'slave'+ ID + ': I am too slow and the master has what it needs'
        
    server_process.terminate()

if __name__ == '__main__':
    main()   
