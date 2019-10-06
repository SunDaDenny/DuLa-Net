import os
import sys

import argparse
import numpy as np

def eval_2d_iou(fp_pred, fp_gt):

    intersect = np.sum(np.logical_and(fp_pred, fp_gt))
    union = np.sum(np.logical_or(fp_pred, fp_gt))

    iou_2d = intersect / union

    return iou_2d

def eval_3d_iou(fp_pred, h_pred, fp_gt, h_gt):

    intersect = np.logical_and(fp_pred, fp_gt)
    
    fp_t_pred = fp_pred - intersect
    fp_t_gt = fp_gt - intersect

    union = fp_t_pred.sum()*h_pred + fp_t_gt.sum()*h_gt + intersect.sum()*max(h_pred,h_gt)
    intersect = intersect.sum()*min(h_pred,h_gt)

    iou_3d = intersect / union

    return iou_3d

def eval_l2(pred, gt):

    return np.sqrt(np.mean(((pred - gt)**2)))