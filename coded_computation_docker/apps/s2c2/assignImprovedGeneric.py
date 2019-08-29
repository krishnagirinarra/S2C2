import os
import sys
import math
import numpy as np

def assignNumChunks(ui, n, k, numRows):
    #Function to assign number of chunks to each node
    chunkAssignmentDict = {}
    decodeInfoDict = {}
    totalSpeed = sum(ui)
    maxChunksPerNode = totalSpeed
    chunks = [0]*totalSpeed
    # Verify Input arguments
    if(maxChunksPerNode==0):
        print("The speeds provided as input are adding up to 0. Please check again")
        sys.exit(1)
    IdxSorted = np.argsort(ui)
    IdxSorted = np.flip(IdxSorted,0)
    if((ui[IdxSorted[0]]/totalSpeed)*k) > 1 :
        print("NOTE: Many nodes are slower. Nodes might not finish their work at same time")
    count = 0
    for u in ui:
        if(u>=1):
            count+=1
    if(count < k):
        print("The number of nodes with <1 speed is less than k. Not feasible")
        sys.exit(1)
    # Algorithm
    Pending = k * totalSpeed
    for Idx in IdxSorted:
        u = ui[Idx]
        #print u, totalSpeed, Pending
        if (u==0):
            numChunks = 0
        else:
            numChunks = np.round(min(((u*1.0)/totalSpeed)*(Pending), maxChunksPerNode))
        #print numChunks
        chunkAssignmentDict[Idx] = int(numChunks)
        Pending -= numChunks #Update remaining chunks
        totalSpeed -= u #Update total speed available
    if (Pending!=0):
        print("ERROR: Few chunks are assigned in <k nodes")
    #print ui
    print chunkAssignmentDict
    return chunkAssignmentDict

def assignChunks(numChunksDict, ui, n, k, numRows):
    #Function to assign exact chunks to each node
    IdxSorted = np.argsort(ui)
    IdxSorted = np.flip(IdxSorted,0)
    maxChunksPerNode = sum(ui)
    #totalChunks = k*maxChunksPerNode
    chunksDict = {}
    chunk = 0
    for Idx in IdxSorted:
        chunkList = [0]*maxChunksPerNode
        for c in xrange(numChunksDict[Idx]):
            chunkList[chunk] = 1
            chunk += 1
            chunk %= maxChunksPerNode
        chunksDict[Idx] = chunkList
    return chunksDict

def convertChunksToRows(chunksDict, ui, numRowsPerNode):
    #Calculate and assign starting row, length for each chunk
    IdxSorted = np.argsort(ui)
    IdxSorted = np.flip(IdxSorted,0)
    maxChunksPerNode = sum(ui)
    rowsDict = {}
    lengthsDict = {}
    decodeInfoDict = {}
    chunkToRow = {}
    chunkToLength = {}
    rowPrev = row = 0
    length = 0
    for i in xrange(maxChunksPerNode+1):
        if (i==maxChunksPerNode):
            length = numRowsPerNode-rowPrev
            chunkToLength[i-1] = length
            break
        row = int(np.round(((i*numRowsPerNode*1.0)/maxChunksPerNode)))
        length = row - rowPrev
        chunkToRow[i] = row
        if (i!=0):
            chunkToLength[i-1] = length
        rowPrev = row
    #print chunkToRow
    #Assign rows, lengths to nodes 
    for Idx in IdxSorted:
        rowsList = [0]*maxChunksPerNode
        lengthsList = [0]*maxChunksPerNode
        for i in xrange(maxChunksPerNode):
            rowsList[i] = chunksDict[Idx][i]*chunkToRow[i] #(1/0) * rowId
            lengthsList[i] = chunksDict[Idx][i]*chunkToLength[i]
        rowsDict[Idx] = rowsList
        lengthsDict[Idx] = lengthsList
    #Create decodeInfo
    for chunk in xrange(maxChunksPerNode):
        decodeInfoDict[chunk] = set()

    for node, assign in chunksDict.iteritems():
        for chunk,val in enumerate(assign):
            if val!=0:
                decodeInfoDict[chunk].add(node)
    
    return rowsDict, lengthsDict, decodeInfoDict

def assignRnLImproved(ui, n, k, numRows):
    chunkMap = []
    numChunksDict = assignNumChunks(ui, n, k, numRows)
    chunksDict = assignChunks(numChunksDict, ui, n, k, numRows)
    rowsDict, lengthsDict, decodeInfoDict = convertChunksToRows(chunksDict, ui, numRows)
    return rowsDict, lengthsDict, decodeInfoDict, chunksDict, chunkMap

if __name__ == '__main__':
    numRows = 840
    n = 12
    k = 6
    uis = []
    uis.append([10, 10, 8, 10, 8, 9, 8, 9, 10, 8, 10, 9])
    uis.append([9, 8, 8, 9, 8, 8, 9, 10, 9, 9, 8, 9])
    uis.append([0, 0, 0, 9, 10, 8, 8, 8, 9, 10, 9, 10])
    uis.append([0, 0, 0, 0, 0, 4, 4, 4, 5, 4, 4, 4])
    uis.append([0, 0, 0, 0, 0, 4, 4, 4, 5, 4, 4, 5])

    for ui in uis:
        #ui = map(lambda x: int(x), np.random.uniform(1,11,5))
        #ui = [9, 8, 8, 9, 8, 8, 9, 10, 9, 9, 8, 9]
        rowsDict, lengthsDict, decodeInfoDict, chunksDict, chunkMap = assignRnLImproved(ui,n,k,numRows)
        print ui
        print chunksDict
        print rowsDict
        print lengthsDict
        print decodeInfoDict
        for de in decodeInfoDict:
            print len(decodeInfoDict[de]),
        print
