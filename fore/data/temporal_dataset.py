### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform
from PIL import Image
import numpy as np
from torch.autograd import Variable
import glob
#import sys
#sys.path.append("../OpticalFlowToolkit/lib/")
#import flowlib as fl

def compute_bbox(mask):
    '''
    :param mask: mask of size(height, width)
    :return: bbox
    '''
    y, x  = np.where(mask == 1)
    if len(x) == 0 or len(y) == 0:
        return None
    bbox = np.zeros((2,2))
    bbox[:,0] = [np.min(x),np.min(y)]
    bbox[:,1] = [np.max(x),np.max(y)]
    return bbox




class TemporalDataset(BaseDataset):
    # Load pre-computed optical flow to save gpu memory
    def initialize(self, opt):
        self.opt = opt
        self.height = int(opt.loadSize/2.0)
        self.width = opt.loadSize
        if opt.isTrain == True:
            self.phase = 'train'
        else:
            self.phase = 'val'
        self.mask_threshold = int(self.height)
        self.all_image_paths = self.load_all_image_paths('./data/cityscapes/tracking/train_data_gen/generate_valid_train_list/dynamic_car/')
        self.n_of_seqs = len(self.all_image_paths)                 # number of sequences to train
        print("Load number of video paths = %d"%self.n_of_seqs)

    def __getitem__(self, index):
        tIn = self.opt.tIn
        tOut = self.opt.tOut
        n_gpu = len(self.opt.gpu_ids)
        tAll = tIn + tOut
        #print(self.all_image_paths[index])
        video_data = self.all_image_paths[index%self.n_of_seqs]
        len_object = len(video_data)
        obj_id = np.random.randint(0, len_object)
        image_paths = video_data[obj_id][0]
        semantic_paths = video_data[obj_id][1]
        back_paths = video_data[obj_id][2]
        masks_paths = video_data[obj_id][3]
        params = get_img_params(self.opt, (self.width, self.height))
        t_bic = get_transform(self.opt, params)
        t_ner = get_transform(self.opt, params, Image.NEAREST, normalize=False)

        Semantics = torch.cat([self.get_image(semantic_paths[p], t_ner, is_label=True) for p in range(tAll)], dim=0)
        Images = torch.cat([self.get_image(image_paths[p], t_bic) for p in range(tAll)], dim=0)
        Masks = torch.cat([self.get_image(masks_paths[p], t_ner) for p in range(tAll)], dim=0)#divide by 255
        Backs = torch.cat([self.get_image(back_paths[p], t_bic) for p in range(tAll)], dim=0)

        # Generate combined images
        Combines = 0
        for i in range(tAll):
            back_path = back_paths[i]
            #print("back_path = ", back_path)
            mask_path = masks_paths[i]
            image_path = image_paths[i]
            back = np.array(Image.open(back_path))
            mask = np.array(Image.open(mask_path)) / 255
            mask = np.tile(np.expand_dims(mask, axis=2), [1,1,3])
            image = np.array(Image.open(image_path))
            #print("image shape = ", image.shape)
            #print("back shape = ", back.shape)
            #print("mask shape =", mask.shape)
            combine = image * mask + back * (1.0 - mask)                        
            Combinei = t_bic(Image.fromarray(combine.astype(np.uint8)))
            Combines = Combinei if i == 0 else torch.cat([Combines, Combinei], dim=0)
           
        # Generate current proposal and current mask
        last_frame_image_path = image_paths[tIn - 1]
        last_frame_object_mask = masks_paths[tIn - 1]
        last_image = np.array(Image.open(last_frame_image_path))
        last_mask = np.array(Image.open(last_frame_object_mask))/ 255
        last_mask = np.tile(np.expand_dims(last_mask, axis=2), [1,1,3])
        last_object = last_image * last_mask
        LastObject = t_bic(Image.fromarray(last_object.astype(np.uint8)))

        return_list = {'Image': Images, 'Back': Backs, 'Mask': Masks, 'Semantic': Semantics, 'Combine': Combines, 'LastObject': LastObject}
        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        #print(A_path)
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255
        return A_scaled

    def __len__(self):
        return self.n_of_seqs

    def name(self):
        return 'TemporalDataset'

    def load_all_image_paths(self, train_list_path):
        video = []
        npy_files = sorted(glob.glob(train_list_path + "*.npy"))
        for i in npy_files:
            video.append(np.load(i, allow_pickle=True))
        return video
        
    def LoadDepthDataSample(self, DepthRoot, images):
        tmp = []
        for p in range(self.tIn):
            curr_full = images[p]
            split_name = curr_full.split("/")
            depth_path = os.path.join(DepthRoot, split_name[-3],split_name[-2],split_name[-1])
            depth_path = depth_path[:-3] + "npy"
            tmp.append(scipy.misc.imresize(np.load(depth_path),(self.h, self.w)))
        Depth = np.concatenate([np.expand_dims(np.expand_dims(tmp[q], 0), 3) for q in range(self.tIn)],axis=3)
        # Depth may be zero, compute average value then 1/average depth
        return Depth

    def IOU_mask(self, mask_A, mask_B):
        #semantic instance,
        mask_A = mask_A.astype(np.bool)
        mask_B = mask_B.astype(np.bool)
        return 1.0 * (mask_A & mask_B).astype(np.int32).sum() / mask_B.astype(np.int32).sum()


    def load_object_mask_val(self, instance_mask_list, images, depth):
        '''
        :param instance: instance contains gt instance or
        :param semantic:
        :param depth:
        :param curr_image:
        :param gt_flag:
        :return:
        '''
        opt = self.opt
        segs = []
        
        for j in range(len(instance_mask_list)):
            #print(instance_mask_list[j])
            cnt_info = []
            flag = True
            for k in range(self.tIn):
                cnt_mask = np.array(Image.open(instance_mask_list[j][k]).resize((self.sp*2, self.sp), resample=Image.NEAREST))/255
                cnt_mask_expand = expand_dims_2(cnt_mask)
                cnt_bbox = compute_bbox(cnt_mask)
                if cnt_bbox is None:
                    continue
                big_bbox = self.enlarge_bbox(cnt_bbox)
                big_mask = self.bbox2mask(big_bbox)
                cnt_bbox_mask = self.bbox2mask(cnt_bbox)
                cnt_depth = np.mean(cnt_mask_expand * self.depthInput[:,:,:,k:k+1])
                cnt_color_image = np.tile(cnt_mask_expand, [1, 1, 1, 3]) * images[:,:,:,k*3:(k+1)*3]
                #scipy.misc.imsave("./debug/segs_%d.png" % j, cnt_color_image[0, :, :, :])
                big_image = np.tile(expand_dims_2(big_mask), [1, 1, 1, 3]) * images[:,:,:,k*3:(k+1)*3]
                cnt_info.append(
                        (cnt_mask, cnt_color_image[0, :, :, :], cnt_depth, cnt_bbox, big_image[0, :, :, :]))
            if len(cnt_info) > 0:
                segs.append(cnt_info)
        return segs


    def preprocess_bike_person(self, instance_list):
        valid_index = np.zeros(len(instance_list)) + 1
        # classes
        #'person'-1, 'bicycle'-2, 'car'-3, 'motorcycle'-4,'bus'-6, 'train'-7, 'truck'-8
        valid_class = [1,2,3,4,6,7,8]
        mask_all = []
        for i in range(len(instance_list)):
            if valid_index[i] == 0:
                continue
            curr_list = instance_list[i]
            if curr_list['class_id'] not in valid_class:
                continue
            if curr_list['class_id'] == 2 or curr_list['class_id'] == 4:
                iou_score = -1
                person_id = -1
                bbox_bike = curr_list['bbox']
                bbox_mask_bike = bbox2mask_maskrcnn(bbox_bike)
                for j in range(len(instance_list)):
                    if valid_index[j] == 1:
                        if instance_list[j]['class_id'] == 1:
                            bbox_person = instance_list[j]['bbox']
                            bbox_mask_person = bbox2mask_maskrcnn(bbox_person)
                            iou = self.IOU_mask(bbox_mask_bike, bbox_mask_person)
                            if iou > iou_score:
                                iou_score = iou
                                person_id = j
                if iou_score > 0:
                    mask_all.append(((curr_list['mask'] | instance_list[person_id]['mask']), 9))
                    valid_index[i] = 0
                    valid_index[person_id] = 0
        for k in range(len(instance_list)):
            if valid_index[k] == 1 and instance_list[k]['class_id'] in valid_class:
                mask_all.append((instance_list[k]['mask'], instance_list[k]['class_id']))
        return mask_all

    def enlarge_bbox(self, bbox):
        '''
        bbox[:, 0] = [np.min(x), np.min(y)]
        bbox[:, 1] = [np.max(x), np.max(y)]
        bbox [min_x, max_x]
             [min_y, max_y]
        '''
        # enlarge bbox to avoid any black boundary
        if self.opt.sp == 256:
            gap = 2
        elif self.opt.sp == 512:
            gap = 4
        elif self.opt.sp == 1024:
            gap = 8
        bbox[0,0] = np.maximum(bbox[0,0] - gap, 0)
        bbox[1,0] = np.maximum(bbox[1,0] - gap, 0)
        bbox[0,1] = np.minimum(bbox[0,1] + gap, self.w-1)
        bbox[1,1] = np.minimum(bbox[1,1] + gap, self.h-1)
        return bbox

    def bbox2mask(self, bbox):
        mask = np.zeros((self.h, self.w))
        bbox = bbox.astype(np.int32)
        mask[bbox[1,0]:bbox[1,1]+1,bbox[0,0]:bbox[0,1]+1] = 1
        return mask
