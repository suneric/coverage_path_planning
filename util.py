import numpy as np
import copy
from map import *
from viewpoint import *

MAXVALUE = 1000000

def vpDistance(vp1,vp2):
    dx = vp1.location[0]-vp2.location[0]
    dy = vp1.location[1]-vp2.location[1]
    dz = vp1.location[2]-vp2.location[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def vpLandOverlapCount(vp1,vp2):
    set1 = set(vp1.gridCover)
    set2 = set(vp2.gridCover)
    intersect = set1 & set2
    return len(intersect)

class ViewPointUtil(object):
    def __init__(self, map, viewpoints, nb=8, overlapRatio=0.3):
        self.map = map
        self.nbCount = nb
        self.viewpoints = viewpoints
        self.nbMap = self.buildNeighborMap(viewpoints,overlapRatio)

    def neighbors(self, vpIdx):
        return self.nbMap[vpIdx][0:self.nbCount]

    """
    build a map for viewpoints
    """
    def buildNeighborMap(self,vps,cn):
        print("=== start building neighbor map ===".format())
        dim = len(vps)
        nbMap = [None]*dim
        for i in range(dim):
            # print("{} / {} viewpoint".format(i+1,dim))
            nbvps = self.searchNeighbor(i,vps,cn)
            nbMap[i] = nbvps
        print("=== end building neighbor map ===".format())
        return nbMap

    """
    search neighbor viewpoints of a given viewpoint
    considering the distance and overlap
    """
    def searchNeighbor(self,i,vps,r):
        dim = len(vps)
        vp = vps[i]
        scoreList = [None]*dim
        for j in range(dim):
            vp1 = vps[j]
            if vp1.id == vp.id:
                scoreList[j] = MAXVALUE
            else:
                dist = vpDistance(vp, vp1)
                overlap = vpLandOverlapCount(vp, vp1)
                scoreList[j]=(1.0-r)*dist + r*overlap
        # return the viewpoint indices with least value
        sortedIndice = np.argsort(np.array(scoreList))
        return sortedIndice
