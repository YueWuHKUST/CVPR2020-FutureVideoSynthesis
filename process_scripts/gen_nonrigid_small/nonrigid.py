import numpy as np
import scipy
import os
from PIL import Image
import glob
import pandas as pd
import cv2


test_list = "../test_list_kitti_myback.npy"

result_path = '/disk1/yue/kitti/non_rigid_mask/val/'

all_instance_list = np.load(test_list, allow_pickle=True)

num = len(all_instance_list)
print("Samples number = ", num)



def add_masks(masks, frames):
    if len(frames) == 4:
        for k in range(4):
            masks[k] += frames[k]
            masks[k] = clip_mask(masks[k])
    elif len(frames) == 1:
        masks[-1] += frames[0]
        masks[-1] = clip_mask(masks[-1])
    else:
        print("Debug here!")
    return masks

def generate_mask(instance_list):
    num = len(instance_list)
    print("number of instances =", num)
    masks = [None]*4
    for i in range(4):
        masks[i] = np.zeros((256, 832))
    for i in range(num):
        cnt_instance = instance_list[i]
        frames = cnt_instance[0]
        if len(frames) == 256:
            frames = [frames]
        if np.sum(frames[-1]) < 0.005 * 256 * 832:
            continue
        cl = cnt_instance[1]
        if cl in [11, 12]:
            masks = add_masks(masks, frames)
    return masks


def clip_mask(mask):
    return np.maximum(np.minimum(mask, 1), 0)

for i in range(num):
    curr_instance = all_instance_list[i][5]
    cnt_num = len(curr_instance)
    non_rigid = generate_mask(curr_instance)
    cnt_result_path = result_path + "%04d/"%i
    if not os.path.exists(cnt_result_path):
        os.makedirs(cnt_result_path)
    for k in range(4):
        tmp = non_rigid[k]*255
        tmp_im = Image.fromarray(tmp.astype(np.uint8)).save(cnt_result_path + "non_rigid_mask_%02d.png"%k)

