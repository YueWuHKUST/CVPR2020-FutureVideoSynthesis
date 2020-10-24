import glob, os
from PIL import Image
import numpy as np
import cv2
root = "/disk2/yue/result/kitti/"
height = 256 # 512 for citysca[es
width = 832 # 1024 for cityscapes
num = 1337
pred_len = 5
def process_fore(fore):
    kernel = np.ones((3,3),np.uint8)
    mask_dilate = cv2.dilate(fore, kernel,iterations = 1)
    mask_erode = cv2.erode(fore, kernel,iterations = 1)
    mask = mask_dilate - mask_erode
    return mask

for i in range(num):
    print(i)
    cnt_root = root + "%04d/"%i
    for j in range(5):
        occ = np.zeros((height, width))
        occ_boundary = np.zeros((height, width)) + 1
        fore_mask = cnt_root + "fore_mask_%02d.png"%j
        if os.path.exists(fore_mask):
            back_mask = cnt_root + "warp_mask_bwd_%02d.png"%j
            fore_arr = np.array(Image.open(fore_mask))/255.0
            fore_boundary = process_fore(fore_arr)
            occ[fore_arr == 1] = 1
            
            back_arr = np.array(Image.open(back_mask))/255.0
            occ[back_arr == 1] = 1
            occ[fore_boundary == 1] = 0
            occ = occ*255

            occ_boundary[fore_boundary == 1] = 0
            occ_boundary = occ_boundary * 255
            Image.fromarray(occ.astype(np.uint8)).save(cnt_root + "occ_%02d.png"%j)
            Image.fromarray(occ_boundary.astype(np.uint8)).save(cnt_root + "occ_boundary_%02d.png"%j)
