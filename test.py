from map import *
from viewpoint import *
from util import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if __name__ == "__main__":
    map = GridMap()
    map.makeMap(20,20,1)
    #map.loadMap(os.path.join(sys.path[0],'./data/5050.txt'))
    vps = generateViewPoints(gridMap = map,fov = (40.0,40.0), resolution=1.0)

    util = ViewPointUtil(map,vps,nb=4,overlapRatio=.9)
    startIdx = np.random.randint(len(vps))
    startVp = vps[startIdx]
    nbvps = [vps[i] for i in util.neighbors(startVp.id)]
    nbvps.insert(0,startVp)

    fig = plt.figure(figsize=(15,12)) # inch
    ax = fig.add_subplot(111)
    map.plotMap(ax)
    plotViewpoints(ax,nbvps,2)
    plt.show()

    #map.saveMap(os.path.join(sys.path[0],'./data/5050.txt'))
