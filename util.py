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
        # self.nbMap = self.buildNeighborMap(viewpoints,overlapRatio)
        self.vpLeft = self.leftvps()
        self.vpRight = self.rightvps()
        self.vpUp = self.upvps()
        self.vpDown = self.downvps()
        self.nbMap = self.buildNeighborMap2d()

    def leftvps(self):
        left = [None]*len(self.viewpoints)
        for vp in self.viewpoints:
            id = vp.id
            if id % int(self.map.width) > 0:
                left[id] = id-1
        return left

    def rightvps(self):
        right = [None]*len(self.viewpoints)
        for vp in self.viewpoints:
            id = vp.id
            if id % int(self.map.width) < int(self.map.width)-1:
                right[id] = id+1
        return right

    def upvps(self):
        up = [None]*len(self.viewpoints)
        for vp in self.viewpoints:
            id = vp.id
            if id < int(len(self.viewpoints)-self.map.width):
                up[id] = id + int(self.map.width)
        return up

    def downvps(self):
        down = [None]*len(self.viewpoints)
        for vp in self.viewpoints:
            id = vp.id
            if id > int(self.map.width):
                down[id] = id - int(self.map.width)
        return down


    def neighbors_2d(self, vpIdx):
        vp = self.viewpoints[vpIdx]

        vpl = vp
        lid = self.vpLeft[vp.id]
        while lid != None:
            vpl = self.viewpoints[lid]
            if vpLandOverlapCount(vp,vpl) > 0:
                lid = self.vpLeft[vpl.id]
            else:
                break

        vpr = vp
        rid = self.vpRight[vp.id]
        while rid != None:
            vpr = self.viewpoints[rid]
            if vpLandOverlapCount(vp,vpr) > 0:
                rid = self.vpRight[vpr.id]
            else:
                break

        vpu = vp
        uid = self.vpUp[vp.id]
        while uid != None:
            vpu = self.viewpoints[uid]
            if vpLandOverlapCount(vp,vpu) > 0:
                uid = self.vpUp[vpu.id]
            else:
                break

        vpd = vp
        did = self.vpDown[vp.id]
        while did != None:
            vpd = self.viewpoints[did]
            if vpLandOverlapCount(vp,vpd) > 0:
                did = self.vpDown[vpd.id]
            else:
                break

        return [vpl.id,vpr.id,vpu.id,vpd.id]

    def buildNeighborMap2d(self):
        dim = len(self.viewpoints)
        nbMap = [None]*dim
        for i in range(dim):
            nbvps = self.neighbors_2d(i)
            nbMap[i] = nbvps
        # print(nbMap)
        return nbMap


    def neighbors(self, vpIdx):
        return self.nbMap[vpIdx]

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
