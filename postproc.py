import os
import sys
import math

import numpy as np
import cv2
from skimage import draw

import Utils
import config as cf

def fit_floorplan(data):

    #Tthresholding by 0.5
    ret, data_thresh = cv2.threshold(data, 0.5, 1,0)
    data_thresh = np.uint8(data_thresh)
    data_img, data_cnt, data_heri = cv2.findContours(data_thresh, 1, 2)

    # Find the the largest connected component and its bounding box
    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    sub_x, sub_y, w, h = cv2.boundingRect(data_cnt[0])
    data_sub = data_thresh[sub_y:sub_y+h,sub_x:sub_x+w]

    # Binary image to a densely sampled piece-wise linear closed loop
    data_img, data_cnt, data_heri = cv2.findContours(data_sub, 1, 2)
    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    epsilon = 0.005*cv2.arcLength(data_cnt[0], True)
    approx = cv2.approxPolyDP(data_cnt[0], epsilon,True)

    # Regression analysis on the edges
    # Cluster them into sets of axis-aligned horizontal and vertical lines
    x_lst = [0,]
    y_lst = [0,]
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i+1)%len(approx)][0]

        if (p2[0]-p1[0]) == 0:
            slope = 10
        else:
            slope = abs((p2[1]-p1[1]) / (p2[0]-p1[0]))
        
        if slope <= 1:
            s = int((p1[1] + p2[1])/2)
            y_lst.append(s)
            
        elif slope > 1:
            s = int((p1[0] + p2[0])/2)
            x_lst.append(s)
            
    x_lst.append(data_sub.shape[1])
    y_lst.append(data_sub.shape[0])
    x_lst.sort()
    y_lst.sort()

    # Merge the points which distance is smaller than the 0.05 * diagonal 
    diag = math.sqrt(math.pow(data_sub.shape[1],2) +  math.pow(data_sub.shape[0],2))

    def merge_near(lst):
        group = [[0,]]
        for i in range(1, len(lst)):
            if lst[i] - np.mean(group[-1]) < diag * 0.05:
                group[-1].append(lst[i])
            else:
                group.append([lst[i],])
        group = [int(np.mean(x)) for x in group]
        return group

    x_lst = merge_near(x_lst)
    y_lst = merge_near(y_lst)
    
    # Divide the bounding rectangle into several disjoint grid cells
    # The 2D floor plan as the union of grid cells where the ratio of floor plan area is 
    # greater than 0.5
    ans = np.zeros((data_sub.shape[0],data_sub.shape[1]))
    for i in range(len(x_lst)-1):
        for j in range(len(y_lst)-1):
            sample = data_sub[y_lst[j]:y_lst[j+1] , x_lst[i]:x_lst[i+1]]            
            score = sample.mean()
            if score >= 0.5:
                ans[y_lst[j]:y_lst[j+1] , x_lst[i]:x_lst[i+1]] = 1

    # Get the final floor plan key points
    pred = np.uint8(ans)
    pred_img, pred_cnt, pred_heri = cv2.findContours(pred, 1, 3)
    pred_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    polygon = [(p[0][1], p[0][0]) for p in pred_cnt[0][::-1]]

    Y = np.array([p[0]+sub_y for p in polygon])
    X = np.array([p[1]+sub_x for p in polygon])
    fp_pts = np.concatenate( (Y[np.newaxis,:],X[np.newaxis,:]), axis=0)

    # Draw the final floor plan map
    fp_pred = np.zeros(data.shape)
    rr, cc = draw.polygon(fp_pts[0], fp_pts[1])
    rr = np.clip(rr, 0, data.shape[0]-1)
    cc = np.clip(cc, 0, data.shape[1]-1)
    fp_pred[rr,cc] = 1

    return fp_pts, fp_pred


def run(fp_prob, fc_prob_up, fc_prob_down, height_pred):

    # Floor-ceiling map(down-view) need to be normalized by the ratio
    scale =  cf.camera_h / (height_pred - cf.camera_h)
    fc_prob_down_r = Utils.resize_crop(fc_prob_down, scale, cf.fp_size)

    # Fused floor plan probability maps
    fp_prob_fuse = fp_prob * 0.5 + fc_prob_up * 0.25 + fc_prob_down_r * 0.25

    # Run 2D floor plan fitting
    fp_pts, fp_pred = fit_floorplan(fp_prob_fuse)

    return fp_pts, fp_pred
