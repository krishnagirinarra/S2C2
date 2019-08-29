import time
import sys
import numpy as np
import math

def generate_data(n, k, r, c, a=3, b=3):
    #(n,k) coding
    #rows and columns
    #a = sub-partitions of A (left side Matrix); b = sub-partitions of B (right side Matrix)
    assert (k==a*b)
    A = np.random.rand(c, r)
    A_T = np.transpose(A)
    encoding_a = [[1,0,0],[1,1,1],[1,2,4]]
    encoding_b = [[1,0,0],[1,1,1],[1,4,16]]
    for idx in range(n):
        #For the case where a = 3
        r_m = int(math.floor(r/3))
        encoded_AT = A_T[:r_m,:] + A_T[r_m:2*r_m,:]*(idx) + A_T[2*r_m:3*r_m,:]*(idx**2)
        #For the case where b = 3
        r_m = int(math.floor(r/3))
        encoded_A = A[:, :r_m] + A[:, r_m:2*r_m]*(idx**2) + A[:, 2*r_m:3*r_m]*(idx**4)
        ptFile_AT = '/home/krishna/data_folder/s2c2/12_9/poly_partition_AT%d.mat' % (idx+1)
        ptFile_A = '/home/krishna/data_folder/s2c2/12_9/poly_partition_A%d.mat' % (idx+1)
        np.savetxt(ptFile_AT, encoded_AT, fmt='%d')
        np.savetxt(ptFile_A, encoded_A, fmt='%d')

if __name__ == "__main__":
    generate_data(12,9, 6000, 6000, 3, 3)
