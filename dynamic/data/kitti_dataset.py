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
from scipy.ndimage.measurements import label
import psutil

class KittiDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.height = 256
        self.width = 512
        self.isTrain = opt.isTrain
        if opt.isTrain == True:
            phase = 'train'
        else:
            phase = opt.phase
        self.all_image_paths = self.load_all_image_paths(opt.ImagesRoot + phase + '/', \
                                               opt.SemanticRoot + phase + '/', \
                                               opt.InstanceRoot + phase + '/', \
                                               opt.DynamicRoot + phase + '/')
        self.n_of_seqs = len(self.all_image_paths)                 # number of sequences to train
        print("Load number of video paths = %d"%self.n_of_seqs)
        self.seq_len_max = max([len(A[0]) for A in self.all_image_paths])


    def __getitem__(self, index):
        tIn = self.opt.tIn
        curr_paths = self.all_image_paths[index]
        image_paths = curr_paths[0]
        semantic_paths = curr_paths[1]
        instance_paths = curr_paths[2]
        
        # setting parameters
        params = get_img_params(self.opt, (self.width, self.height))
        #print("params = ", params)
        transform_scale_image = get_transform(self.opt, params)
        transform_scale_semantic = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        # read in images
        Images = 0
        Semantics = 0
        Instancs = 0
        for i in range(tIn):
            image_path = image_paths[i]
            semantic_path = semantic_paths[i]
            instance_path = instance_paths[i]
            semantic_numpy = self.get_image_numpy(semantic_path)        
            Semantici = self.numpy2tensor(semantic_numpy, transform_scale_semantic, True)
            Imagei = self.get_image_test(image_path, transform_scale_image)
            Instancei = self.get_image(instance_path, transform_scale_semantic, True)
            Images = Imagei if i == 0 else torch.cat([Images, Imagei], dim=0)
            Semantics = Semantici if i == 0 else torch.cat([Semantics, Semantici], dim=0)
            Instancs = Instancei if i == 0 else torch.cat([Instancs, Instancei], dim=0)
        #print(dynamic_paths)
        if self.isTrain is True:
            dynamic_paths = curr_paths[3]
            dynamic = self.load_dynamic(dynamic_paths[0], transform_scale_semantic)
        
            return_list = {'Image': Images, 'Semantic': Semantics, 'Instance': Instancs, 'Dynamic': dynamic, \
                       'Image_path': image_paths, 'Semantic_path': semantic_paths}
        else:
            return_list = {'Image': Images, 'Semantic': Semantics, 'Instance': Instancs}
        return return_list


    def load_dynamic(self, path, transform_scaleA):
        img = Image.open(path)
        #if img.size[1] != self.height:
        #    img = img.resize((self.width, self.height), resample=Image.NEAREST)
        img = np.array(img)
        mask = img == 151
        mask = mask.astype(np.uint8)
        mask_scaled = transform_scaleA(Image.fromarray(mask))
        mask_scaled *= 255
        return mask_scaled #torch.from_numpy(mask)

    def get_image(self, A_path, transform_scaleA, is_label=False):
        if os.path.exists(A_path):
            A_img = Image.open(A_path)
        else:
            arr = np.zeros((self.height, self.width)) + 255
            A_img = Image.fromarray(arr) 
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def numpy2tensor(self, arr, transform_scaleA, is_label=False):
        A_scaled = transform_scaleA(arr)
        if is_label:
            A_scaled *= 255
        return A_scaled

    def get_image_numpy(self, path):
        img = Image.open(path)
        #if img.size[1] != self.height:
        #    img = img.resize((self.width, self.height), resample=Image.NEAREST)
        return img

    def get_static(self, path):
        static = Image.open(path)
        if static.size[1] != self.height:
            static = static.imresize((self.width, self.height), resample=Image.NEAREST)
        static = np.array(static)/255.0
        return static

    def get_image_test(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)        
        if A_img.size[1] != self.height:
            A_img = A_img.resize((self.width, self.height))
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def __len__(self):
        return self.n_of_seqs

    def name(self):
        return 'KittiDataset'

    def load_all_image_paths(self, image_root, semantic_root, instance_root, dynamic_root):
        '''
        in train mode, load KittiMotion dataset
        '''
        if self.isTrain is True:
            train_set = []
            scene_dir = os.listdir(dynamic_root)
            scene_dir.sort()
            scene_dir = scene_dir
            for i in range(len(scene_dir)):
                label_dir = dynamic_root + scene_dir[i] + "/" + "label/"
                # Load image/semantic/instance dir
                image_dir = image_root + scene_dir[i] + "/image_02/data/"
                semantic_dir = semantic_root + "gray/" + scene_dir[i] + "/image_02/data_transformed/"
                instance_dir = instance_root + scene_dir[i] + "/"
                label_list = os.listdir(label_dir)
                label_list.sort()
                for j in range(len(label_list)):
                    label_full_path = label_dir + label_list[j]
                    # Append 4 image/semantic/instance

                    # Image
                    image_set = []
                    semantic_set = []
                    instance_set = []
                    idx = int(label_list[j][:-4])#idx of label image
                    if idx - 3 < 0:
                        continue
                    for k in range(idx - 3, idx + 1):
                        image_full_path = image_dir + "0%09d.png"%k
                        #print(image_full_path)
                        assert os.path.isfile(image_full_path)
                        image_set.append(image_full_path)
                        
                        semantic_full_path = semantic_dir + "0%09d.png"%k
                        assert os.path.isfile(semantic_full_path)
                        semantic_set.append(semantic_full_path)

                        instance_full_path = instance_dir + "0%09d.png"%k
                        assert os.path.isfile(instance_full_path)
                        instance_set.append(instance_full_path)
                    train_set.append((image_set, semantic_set, instance_set, [label_full_path]))
            return train_set
        else:
            # In test mode, inference pretrained model on kitti whole dataset   
            ### Load all the examples in the dataset
            data_set = []
            scene_dir = os.listdir(image_root)
            scene_dir.sort()
            for i in range(len(scene_dir)):
                image_dir = image_root + scene_dir[i] + "/image_02/data/"
                semantic_dir = semantic_root + "gray/" + scene_dir[i] + "/image_02/data_transformed/"
                instance_dir = instance_root + scene_dir[i] + "/"
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
                    data_set.append((images, semantics, instances))
            return data_set
                
            
        

