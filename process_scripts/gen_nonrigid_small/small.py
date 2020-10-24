import numpy as np
import scipy
import os
from PIL import Image
import glob
import pandas as pd
import cv2


test_list = "../test_list_kitti.npy"

result_path = '/disk1/yue/kitti/small_object_mask/val/'

all_instance_list = np.load(test_list, allow_pickle=True)

num = len(all_instance_list)
print("Samples number = ", num)

kernel = np.ones((3,3),np.uint8)

def add_masks(masks, frames):
    if len(frames) == 4:
        for k in range(4):
            cnt_mask = cv2.dilate(frames[k].astype(np.float32), kernel, iterations=1)
            masks[k] += cnt_mask
            masks[k] = clip_mask(masks[k])
    elif len(frames) == 1:
        cnt_mask = cv2.dilate(frames[0].astype(np.float32), kernel, iterations=1)
        masks[-1] += cnt_mask
        masks[-1] = clip_mask(masks[-1])
    else:
        print("Debug here!")
    return masks

def mask2bbox(mask):
    y, x = np.where(mask == 1)
    min_y = np.min(y)
    max_y = np.max(y)
    min_x = np.min(x)
    max_x = np.max(x)
    center_x = min_x + (max_x - min_x)/2.0
    center_y = min_y + (max_y - min_y)/2.0
    return center_x, center_y

def whether_move(masks, frames):
    if len(frames) == 4:
        c_x_list = [None]*4
        c_y_list = [None]*4
        diff_c_x = [None]*3
        diff_c_y = [None]*3
        for k in range(4):
            cnt_mask = frames[k]
            c_x, c_y = mask2bbox(cnt_mask)
            c_x_list[k] = c_x
            c_y_list[k] = c_y
        for k in range(3):
            diff_c_x[k] = np.abs(c_x_list[k+1] - c_x_list[k])
            diff_c_y[k] = np.abs(c_y_list[k+1] - c_y_list[k])
        if np.maximum(np.array(diff_c_x).max(), np.array(diff_c_y).max()) < 10:
            ### Update Masks
            kernel = np.ones((3,3),np.uint8)
            for k in range(4):
                cnt_mask = cv2.dilate(frames[k].astype(np.float32), kernel, iterations=1)
                masks[k] += cnt_mask
                masks[k] = clip_mask(masks[k])
    return masks


def generate_mask(instance_list):
    num = len(instance_list)
    #print("number of instances =", num)
    masks = [None]*4
    for i in range(4):
        masks[i] = np.zeros((256, 832))
    for i in range(num):
        cnt_instance = instance_list[i]
        frames = cnt_instance[0]
        if len(frames) == 256:
            frames = [frames]
        if np.sum(frames[-1]) < 0.002 * 256 * 832:
            masks = add_masks(masks, frames)
        masks = whether_move(masks, frames)
        ###### Analysis whether this object is moving

    return masks


def clip_mask(mask):
    return np.maximum(np.minimum(mask, 1), 0)


### 00
for i in range(num):
    print(i)
    curr_instance = all_instance_list[i][5]
    cnt_num = len(curr_instance)
    non_rigid = generate_mask(curr_instance)
    #cnt_result_path = result_path + "%04d/"%(i+870)
    cnt_result_path = result_path + "%04d/"%(i)
    if not os.path.exists(cnt_result_path):
        os.makedirs(cnt_result_path)
    for k in range(4):
        tmp = non_rigid[k]*255
        tmp_im = Image.fromarray(tmp.astype(np.uint8)).save(cnt_result_path + "small_object_mask_%02d.png"%k)

