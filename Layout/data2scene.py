import objs
import utils
import config as cf


def pts2scene(fp_pts, height):

    camera_h = cf.camera_h
    scale = (height - camera_h) / camera_h

    fp_pts = fp_pts.astype(float)
    fp_pts -= cf.fp_size / 2
    fp_pts *= scale
    fp_pts = fp_pts.astype(int)

    scene = objs.Scene()
    scene.cameraHeight = camera_h
    scene.layoutHeight = float(height)

    scene.layoutPoints = []
    for i in range(fp_pts.shape[1]):
        fp_xy = fp_pts[:,i] * (cf.fp_meter / cf.fp_size)
        xyz = (fp_xy[1], 0, fp_xy[0])
        scene.layoutPoints.append(objs.GeoPoint(xyz))
    
    scene.genLayoutWallsByPoints(scene.layoutPoints)
    scene.updateLayoutGeometry()

    return scene
    

def json2scene(json):

    scene = objs.Scene()
    utils.loadLabelByJson(json, scene)

    return scene