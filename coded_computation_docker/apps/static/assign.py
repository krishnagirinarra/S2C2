#!/usr/bin/python
## Inputs ##
# ui: list containing speeds of N servers 
# N: Number of servers
# K: (N,K)MDS coding
## Outputs ##
# r: list of starting row numbers
# l: list of number of rows to execute
import sys
import math
"""
if(len(sys.argv)!=3):
    print("Wrong input arguments. Expected: <N: Number of Servers> <K: (N,K) MDS Coding>")
    sys.exit(0)
n = int(sys.argv[1])
k = int(sys.argv[2])
# Ideally need to figure out ui by query of slave nodes
ui = [2, 2, 1, 1]
numRows = 100
"""
def bucketize(ui,n,k): # Divide the n servers into k buckets of near equal loads
    sortedUi = sorted(ui,reverse=True)
    b = []
    load = [0]*k
    tempk = 0
    for i in xrange(n):
        #print i,tempk
        b.append(tempk)
        insertLoad = sortedUi[i]+load[tempk]
        load[tempk] = insertLoad
        tempk = nextk(load)
    return b, load

def nextk(load):
    minLoad = min(load)
    index = load.index(minLoad)
    return index

def assignRnL(ui, n, k, numRows):
    sortedUi = sorted(ui,reverse=True)
    buckets, load = bucketize(ui,n,k)
    r = [0]*k
    l = [0]*k
    rows = [0]*n
    lengths = [0]*n
    for i in xrange(n):
        tempk = buckets[i]
        r[tempk] = r[tempk] + l[tempk]
        templ = (float(sortedUi[i]))/load[tempk]
        templ = int(math.ceil(numRows * templ))
        if (l[tempk] + templ > numRows):
            templ = numRows - l[tempk] 
        l[tempk] = l[tempk] + templ
        rows[i] = r[tempk]
        lengths[i] = templ 
    return rows, lengths 

if __name__ == '__main__':
    #ui = [2, 2, 1, 1]
    ui = [1, 1, 1, 1]
    numRows = 300
    n = 4
    k = 3 
    rows,lengths = assignRnL(ui, n, k, numRows)
    print rows
    print lengths
