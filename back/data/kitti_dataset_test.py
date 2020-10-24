### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform
import scipy
import numpy as np
from torch.autograd import Variable
from PIL import Image
import cv2
import psutil

##### Kitti dataset for background prediction 


class TestKittiDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.height = 256
        self.width = 832
        self.isTrain = opt.isTrain
        self.tIn = self.opt.tIn
        self.tOut = self.opt.tOut
        #if static, use pre-computed static map
        #if not, use background mask
        if self.isTrain is True:
            phase = 'train'
        else:
            phase = 'val'
        self.static = opt.static
        self.all_image_paths = self.load_all_image_paths(opt.ImagesRoot + phase + "/", \
                                               opt.SemanticRoot + phase + "/gray/",\
                                               opt.InstanceRoot + phase + "/", \
                                               opt.StaticMapDir + phase + "/")
        self.n_of_seqs = len(self.all_image_paths)                 # number of sequences to train
        print("Load number of video paths = %d"%self.n_of_seqs)
        self.seq_len_max = 9
        self.n_frames_total = 9

    def __getitem__(self, index):
        tIn = self.tIn
        tOut = self.tOut
        image_paths = self.all_image_paths[index % self.n_of_seqs][0]
        semantic_paths = self.all_image_paths[index % self.n_of_seqs][1]
        instance_paths = self.all_image_paths[index % self.n_of_seqs][2]
    
        if self.static is True:
            static_paths = self.all_image_paths[index % self.n_of_seqs][3]
            nonrigid_paths = self.all_image_paths[index % self.n_of_seqs][4]
            small_paths =  self.all_image_paths[index % self.n_of_seqs][5]
        else:
            back_paths = self.all_image_paths[index % self.n_of_seqs][3]
        # setting parameters
        tAll = tIn
        # setting transformers
        self.origin_w, self.origin_h = Image.open(image_paths[0]).size

        params = get_img_params(self.opt, (self.origin_w, self.origin_h))
        transform_scale_bicubic = get_transform(self.opt, params)
        transform_scale_nearest = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        Images = 0
        Semantics = 0
        Instancs = 0
        Back_mask = 0

        for i in range(tAll):
            image_path = image_paths[i]
            semantic_path = semantic_paths[i]
            instance_path = instance_paths[i]
            nonrigid_path = nonrigid_paths[i]
            small_path = small_paths[i]

            semantic_PIL  = self.get_resize_PIL(semantic_path, Image.NEAREST)
            Semantici = self.numpy2tensor(semantic_PIL, transform_scale_nearest, True)
            image_PIL = self.get_resize_PIL(image_path)
            #Imagei = self.numpy2tensor(image_PIL, transform_scale_bicubic)
            ### Only load input static maps
            if self.static:
                non_rigid = self.read_non_rigid(nonrigid_path)
                small = self.read_non_rigid(small_path)
                #print(small_path)
                # Load static
                static_path = static_paths[i]
                
                static_PIL = self.get_resize_PIL(static_path)
                static = self.delete_non_rigid(static_PIL, non_rigid, small)
                # Load image
                Imagei = self.get_image_back(image_PIL, transform_scale_bicubic, static)#
                statici = self.mask2tensor(static, transform_scale_nearest)
                Back_mask = statici if i == 0 else torch.cat([Back_mask, statici], dim=0)
            else:
                back = self.compute_back(semantic_PIL)
                Imagei = self.get_image_back(image_PIL, transform_scale_bicubic, back)
                backmaski = self.mask2tensor(back, transform_scale_nearest)
                Back_mask = backmaski if i == 0 else torch.cat([Back_mask, backmaski], dim=0)
            
            instance_pil = self.get_resize_PIL(instance_path)
            Instancei = self.mask2tensor(instance_pil, transform_scale_nearest)
            Images = Imagei if i == 0 else torch.cat([Images, Imagei], dim=0)
            Semantics = Semantici if i == 0 else torch.cat([Semantics, Semantici], dim=0)
            Instancs = Instancei if i == 0 else torch.cat([Instancs, Instancei], dim=0)
        return_list = {'Image': Images, 'Semantic': Semantics, 'Instance': Instancs, \
                        'back_mask': Back_mask, 
                        #'segsmask': seg_mask, \#'Edge': Edges,  \
                       'Image_path': image_paths, 'Semantic_path': semantic_paths}
        return return_list

    def numpy2tensor(self, arr, transform_scaleA, is_label=False):
        A_scaled = transform_scaleA(arr)
        if is_label:
            A_scaled *= 255
        return A_scaled

    def get_resize_PIL(self, path, method=Image.BICUBIC):
        img = Image.open(path)

        if img.size[0] != self.width:
            img = img.resize((self.width, self.height), resample=method)
        return img

    def get_PIL(self, path, convert=False):
        img = Image.open(path)
        if convert is True:
            img = img.convert('RGB')
        #if img.size[1] != self.origin_h:# resize dynamic masks
        #    img = img.resize((self.origin_w, self.origin_h), Image.NEAREST)
        return img

    def read_non_rigid(self, non_rigid_path):
        mask = np.array(Image.open(non_rigid_path).resize((self.width, self.height), resample=Image.NEAREST))/255
        return mask

    def delete_non_rigid(self, static_pil, non_rigid, small):
        static = 1.0 - np.array(static_pil)/255.0
        static[non_rigid == 1] = 0
        static[small == 1] = 1
        return static
    
    def get_image_back(self, image_pil, transform_scaleA, backmask):
        A_img = np.array(image_pil)*np.tile(np.expand_dims(backmask, axis=2), [1,1,3])
        A_img = Image.fromarray(A_img.astype(np.uint8))
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def compute_non_rigid(self, semantic_PIL):
        semantic_npy = np.array(semantic_PIL)
        non_rigid = np.zeros_like(semantic_npy)   
        non_rigid_idx = [11, 12, 17, 18]
        for b in range(len(non_rigid_idx)):
            non_rigid[semantic_npy == non_rigid_idx[b]] = 1
        return non_rigid

    def compute_back(self, arr):
        semantic = np.array(arr)
        back = semantic < 11
        return back.astype(np.int32)

    def mask2tensor(self, mask, transform_scaleA):
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        mask_tensor = transform_scaleA(mask)
        mask_tensor *= 255
        return  mask_tensor

    def __len__(self):
        return self.n_of_seqs

    def name(self):
        return 'TestKittiDataset'

    def load_all_image_paths(self, image_root, semantic_root, instance_root, static_map_dir):
        non_rigid_dir = self.opt.non_rigid_dir
        small_object_mask_dir = self.opt.small_object_mask_dir
        video = []
        scene_dir = os.listdir(image_root)
        scene_dir.sort()
        video_cnt = 0
        for i in range(len(scene_dir)):
            image_dir = image_root + scene_dir[i] + "/image_02/data/"
            semantic_dir = semantic_root + scene_dir[i] + "/image_02/data_transformed/"
            instance_dir = instance_root + scene_dir[i] + "/"
            #print(instance_dir)
            #back_dir = static_map_dir + "test_latest_kitti" + scene_dir[i] + "/"
            image_list = os.listdir(image_dir)
            image_list.sort()
            semantic_list = os.listdir(semantic_dir)
            semantic_list.sort()
            instance_list = os.listdir(instance_dir)
            instance_list.sort()
            for k in range(len(image_list)-8):
                images = []
                semantics = []
                instances = []
                for f in range(k, k + 4):
                    image_full_path = image_dir + image_list[f]
                    assert os.path.isfile(image_full_path)
                    images.append(image_full_path)
                    semantic_full_path = semantic_dir + semantic_list[f]
                    #print(semantic_full_path)
                    assert os.path.isfile(semantic_full_path)
                    semantics.append(semantic_full_path)
                    #print(instance_dir + instance_list[f])
                    instance_full_path = instance_dir + instance_list[f]
                    assert os.path.isfile(instance_full_path)
                    instances.append(instance_full_path)

                #static_curr_dir = back_dir + "%04d/"%k
                static_maps = []
                non_rigid_paths = []
                small_object_mask = []
                for p in range(4):
                    static_full_path = static_map_dir + "%04d/"%video_cnt + "pred_dynamic_%02d.png"%p
                    #print(static_full_path)
                    assert os.path.isfile(static_full_path)
                    static_maps.append(static_full_path)

                    full_nonrigid_path = non_rigid_dir + "%04d/"%video_cnt + "non_rigid_mask_%02d.png"%(p)
                    assert os.path.isfile(full_nonrigid_path)
                    non_rigid_paths.append(full_nonrigid_path)

                    full_small_object_path = small_object_mask_dir + "%04d/"%video_cnt + "small_object_mask_%02d.png"%(p)
                    assert os.path.isfile(full_small_object_path)
                    small_object_mask.append(full_small_object_path)
                video.append((images, semantics, instances, static_maps, non_rigid_paths, small_object_mask))
                video_cnt = video_cnt + 1
        return video
