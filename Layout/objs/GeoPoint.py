import utils 

gpInstanceCount = 0

class GeoPoint(object):

    def __init__(self, data):

        if(len(data)==2):
            self.coords = data
            self.xyz = utils.coords2xyz(self.coords, 1)
        else:
            self.xyz = data
            self.coords = utils.xyz2coords(self.xyz)

        global gpInstanceCount
        gpInstanceCount += 1
        self.id = gpInstanceCount

    def moveByVector(self, vec):

        self.xyz = utils.vectorAdd(self.xyz, vec)
        self.coords = utils.xyz2coords(self.xyz)