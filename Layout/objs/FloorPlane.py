import objs
import utils 

fpInstanceCount = 0

class FloorPlane(object):

    def __init__(self, scene, isCeiling=False):

        self.scene = scene
        self.isCeiling = isCeiling

        self.gPoints = scene.layoutPoints
        self.walls = scene.layoutWalls
        self.color = (0, 0, 0)
        
        self.normal = (0, -1, 0) if isCeiling else (0, 1, 0)
        self.height = 0
        self.planeEquation = (0, 0, 0, 0)

        self.corners = []
        self.edges = []

        self.init()

        global fpInstanceCount
        fpInstanceCount += 1
        self.id = fpInstanceCount

    def init(self):
    
        self.updateGeometry()
    
    def updateGeometry(self):

        cameraH = self.scene.cameraHeight
        cam2ceilH =  self.scene.layoutHeight - cameraH 
        self.height = cam2ceilH if self.isCeiling else cameraH 
        self.planeEquation = self.normal + (self.height,)
        self.color = utils.normal2color(self.normal)

        self.updateCorners()
        self.updateEdges()
        
    def updateCorners(self):

        self.corners = []
        for gp in self.gPoints:
            if self.isCeiling:
                xyz = (gp.xyz[0], self.height, gp.xyz[2])
            else:
                xyz = (gp.xyz[0], -self.height, gp.xyz[2])
            corner = objs.GeoPoint(xyz)
            self.corners.append(corner)
    
    def updateEdges(self):
        
        self.edges = []
        cnum = len(self.corners)
        for i in range(cnum):
            edge = objs.GeoEdge(self.corners[i], self.corners[(i+1)%cnum])
            self.edges.append(edge)
