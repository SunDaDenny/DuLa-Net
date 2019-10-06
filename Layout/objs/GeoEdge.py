import utils 

geInstanceCount = 0

class GeoEdge(object):

    def __init__(self, gPoint1, gPoint2):

        self.gPoints = (gPoint1, gPoint2)
        self.vector = utils.pointsDirection(gPoint1.xyz, gPoint2.xyz)
        self.sample = utils.pointsSample(gPoint1.xyz, gPoint2.xyz, 30)
        self.coords = utils.points2coords(self.sample)

        global geInstanceCount
        geInstanceCount += 1
        self.id = geInstanceCount
    
    def checkCross(self):
        for i in range(len(self.coords)-1):
            isCross, l, r = utils.pointsCrossPano(self.sample[i],
                                                 self.sample[i+1])
            if isCross:
                return True, l, r
        return False, None, None