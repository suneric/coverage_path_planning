#
import matplotlib

"""
SquareGrid
anchor: the base point of the square
length: side length of the square
id: unique id
status: 0 or 1, for the grid is valid on map
cover: covered by a view: 0, no covering, n covered by n views
"""
class SquareGrid:
    def __init__(self, anchor, length, id):
        self.id = id
        self.anchor = anchor # bottom left position [width, height]
        self.length = length

    # check in a pt is in grid
    def inGrid(self, pt):
        if pt[0] < self.anchor[0] or pt[0] > self.anchor[0]+self.length:
            return False
        if pt[1] < self.anchor[1] or pt[1] > self.anchor[1]+self.length:
            return False
        return True

    def center(self):
        return (self.anchor[0]+0.5*self.length, self.anchor[1]+0.5*self.length)

"""
GridMap
"""
class GridMap:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.grids = []

    def makeMap(self, width = 90, height = 60, res=1, sn=1000):
        self.height = height
        self.width = width
        self.grids = self.makeGrids(res)
        print("make map ({} meters x {} meters) width {} grids in total.".format(self.width, self.height, len(self.grids)))

    def makeGrids(self, res=1.0):
        grids = []
        nrow = int(self.height/res)
        ncol = int(self.width/res)
        for i in range(nrow):
            for j in range(ncol):
                sg = SquareGrid(anchor=(j*res,i*res), length=res, id=j+i*ncol)
                grids.append(sg)
        return grids

    def loadMap(self,filename):
        with open(filename, 'r') as reader:
            lines = reader.read().splitlines()
            for i in range(len(lines)):
                data = lines[i].split(" ")
                if i == 0:
                    self.height = float(data[0])
                    self.width = float(data[1])
                else:
                    id = int(data[0])
                    anchor = [float(data[1]),float(data[2])]
                    length = float(data[3])
                    grid = SquareGrid(anchor,length,id)
                    self.grids.append(grid)
        reader.close()
        print("load map ({} meters x {} meters) width {} grids in total.".format(self.width, self.height, len(self.grids)))

    def saveMap(self,filename):
        with open(filename,'w') as writer:
            writer.write(str(self.height) + " " + str(self.width) + "\n")
            for i in range(len(self.grids)):
                grid = self.grids[i]
                line = str(grid.id) + " "\
                     + str(grid.anchor[0]) + " "\
                     + str(grid.anchor[1]) + " "\
                     + str(grid.length) + "\n"
                writer.write(line)
            writer.close()

    def plotMap(self, ax, plotSeeds = False):
        ax.autoscale(enable=False)
        ax.set_xlim([-10,self.width+10])
        ax.set_ylim([-10,self.height+10])

        for grid in self.grids:
            patch = matplotlib.patches.Rectangle((grid.anchor),grid.length, grid.length, facecolor = "green", edgecolor='black',linewidth=1.0,alpha=0.2)
            ax.add_patch(patch)
