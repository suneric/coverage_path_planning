import math
import random
import numpy as np
import matplotlib

"""
A ViewPoint is identifed with its location (x,y,z,yaw) and the sensor FOV(angle1,angle2)
yaw (about z axis), angle1 (in x direction) an angle2 (in y direction) are measured in degree
"""
class ViewPoint:
    def __init__(self,location=(0.0,0.0,1.0,90.0),fov=(60.0,60.0),id=0):
        self.location = location
        self.fov = fov
        self.id = id
        self.view = self.coverArea()
        self.gridCover = []

    def coverArea(self):
        """
        cover area is calculated with give the working distance (z) and the FOV
        return a rectangle vertices in np.array [x,y,0]
        """
        center = (self.location[0],self.location[1])
        fov1 = math.radians(0.5*self.fov[0])
        fov2 = math.radians(0.5*self.fov[1])
        xlen = self.location[2]*np.tan(fov1)
        ylen = self.location[2]*np.tan(fov2)
        xmin = center[0] - xlen
        xmax = center[0] + xlen
        ymin = center[1] - ylen
        ymax = center[1] + ylen

        yaw = math.radians(self.location[3])
        rmat = np.matrix([[np.cos(yaw),-np.sin(yaw),0],
                          [np.sin(yaw),np.cos(yaw),0],
                          [0,0,1]])

        pt0 = self.rotatePoint(center,yaw,(xmin,ymin))
        pt1 = self.rotatePoint(center,yaw,(xmax,ymin))
        pt2 = self.rotatePoint(center,yaw,(xmax,ymax))
        pt3 = self.rotatePoint(center,yaw,(xmin,ymax))
        return (pt0,pt1,pt2,pt3)

    def rotatePoint(self,center,angle,p):
        s = np.sin(angle)
        c = np.cos(angle)
        x = p[0]-center[0]
        y = p[1]-center[1]
        xnew = x*c-y*s + center[0]
        ynew = x*s+y*c + center[1]
        return (xnew,ynew)

    def plotView(self,ax,type=0):
        if type == 0: # viewpoint
            x,y = self.location[0], self.location[1]
            ax.scatter([x],[y], s=1, c='red', marker='o')
        elif type == 1: # boundary
            x = [self.view[0][0],self.view[1][0],self.view[2][0],self.view[3][0],self.view[0][0]]
            y = [self.view[0][1],self.view[1][1],self.view[2][1],self.view[3][1],self.view[0][1]]
            ax.plot(x,y,linewidth=2,color="red")
        else: # coverage
            for grid in self.gridCover:
                patch = matplotlib.patches.Rectangle((grid.anchor),grid.length, grid.length, facecolor = "red", edgecolor='black',linewidth=1.0,alpha=0.2)
                ax.add_patch(patch)



"""
Generate viewpoints
gridMap: the map with grid information
workingDistance: the distance from the ground to the camera
type: "random": randomly generated the viewpoints, "uniform": each grid will have a viewpoint above on it.
return a list of viewpoints
"""
def generateViewPoints(gridMap, fov, resolution = 2.0, workingDistance = 3, type = "uniform"):
    vps = []
    grids = gridMap.makeGrids(resolution) # make another grids from the map for generating viewpoint
    size = len(grids)
    for c in range(size):
        base = (0,0)
        if type == "random":
            base = (random.randrange(gridMap.width), random.randrange(gridMap.height))
        else:
            base = grids[c].center()

        # create a viewpoint with given distance and rotation angle 0.0
        pose = (base[0], base[1], workingDistance, 0.0)
        vp = ViewPoint(location=pose, fov = fov, id=c)
        vps.append(vp)

    for vp in vps:
        computeViewCoverGrids(gridMap.grids, vp)

    print("generate {} viewpoints with resolution {:.2f} at working distance {:.2f}".format(len(vps), resolution, workingDistance))
    return vps

"""
compute the covering grid under a viewpoint
"""
def computeViewCoverGrids(grids, viewPt):
    xmin,xmax,ymin,ymax = viewPt.view[0][0],viewPt.view[1][0],viewPt.view[1][1],viewPt.view[2][1]
    for grid in grids:
        if grid.anchor[0] > xmax or grid.anchor[0]+grid.length < xmin:
            continue
        if grid.anchor[1] > ymax or grid.anchor[1]+grid.length < ymin:
            continue

        viewPt.gridCover.append(grid)
    return


def loadViewpoints(filename, map):
    vps = []
    with open(filename,'r') as reader:
        for line in reader.read().splitlines():
            sections = line.split(",")
            data = sections[0].split(" ")
            id = int(data[0])
            location = (float(data[1]), float(data[2]), float(data[3]), float(data[4]))
            fov = (float(data[5]), float(data[6]))
            vp = ViewPoint(location,fov,id)
            gridIndices = sections[1].split(" ")
            for id in gridIndices:
                vp.gridCover.append(map.grids[int(id)])
            vps.append(vp)
    reader.close()
    print("load {} viewpoints".format(len(vps)))
    return vps

def saveViewpoints(filename, vps):
    with open(filename, 'w') as writer:
        for vp in vps:
            line = str(vp.id) + " "\
                 + str(vp.location[0]) + " " + str(vp.location[1]) + " " + str(vp.location[2]) + " " + str(vp.location[3]) + " "\
                 + str(vp.fov[0]) + " " + str(vp.fov[1]) + ","
            for i in range(len(vp.gridCover)-1):
                line += str(vp.gridCover[i].id) + " "
            line += str(vp.gridCover[len(vp.gridCover)-1].id)
            line +="\n"
            writer.write(line)
    writer.close()
    return

def plotViewpoints(ax, vps, type=0):
    for vp in vps:
        vp.plotView(ax,type)
    return
