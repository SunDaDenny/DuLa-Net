import numpy as np
import cv2

def resize_crop(img, scale, size):
    
    re_size = int(img.shape[0]*scale)
    img = cv2.resize(img, (re_size, re_size), cv2.INTER_AREA)

    if size <= re_size:
        pd = int((re_size-size)/2)
        img = img[pd:pd+size,pd:pd+size]
    else:
        new = np.zeros((size,size))
        pd = int((size-re_size)/2)
        new[pd:pd+re_size,pd:pd+re_size] = img[:,:]
        img = new

    return img

def var2np(data_lst):

    def trans(data):
        if data.shape[1] == 1:
            return data[0, 0].data.cpu().numpy()
        elif data.shape[1] == 3: 
            return data[0, :, :, :].permute(1, 2, 0).data.cpu().numpy()

    if isinstance(data_lst, list):
        np_lst = []
        for data in data_lst:
            np_lst.append(trans(data))
        return np_lst
    else:
        return trans(data_lst)