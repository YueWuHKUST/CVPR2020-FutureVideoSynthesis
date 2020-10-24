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

class TemporalDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.height = int(opt.loadSize/2.0)
        self.width = opt.loadSize
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
        self.n_frames_total = self.opt.n_frames_total      # current number of frames to train in a single iteration

    def __getitem__(self, index):
        tIn = self.opt.tIn
        curr_paths = self.all_image_paths[index]
        image_paths = curr_paths[0]
        semantic_paths = curr_paths[1]
        instance_paths = curr_paths[2]
        dynamic_paths = curr_paths[3]
        # setting parameters
        if self.isTrain is True:
            tAll = tIn
            start_idx = 16
        else:
            tAll = 30
            start_idx = 0
        params = get_img_params(self.opt, (self.width, self.height))
        #print("params = ", params)
        transform_scale_image = get_transform(self.opt, params)
        transform_scale_semantic = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        # read in images
        Images = 0
        Semantics = 0
        Instancs = 0
        for i in range(tAll):
            image_path = image_paths[start_idx + i]
            semantic_path = semantic_paths[start_idx + i]
            instance_path = instance_paths[start_idx + i]
            semantic_numpy = self.get_image_numpy(semantic_path)        
            Semantici = self.numpy2tensor(semantic_numpy, transform_scale_semantic, True)
            Imagei = self.get_image_test(image_path, transform_scale_image)
            Instancei = self.get_image(instance_path, transform_scale_semantic, True)
            Images = Imagei if i == 0 else torch.cat([Images, Imagei], dim=0)
            Semantics = Semantici if i == 0 else torch.cat([Semantics, Semantici], dim=0)
            Instancs = Instancei if i == 0 else torch.cat([Instancs, Instancei], dim=0)
        #print(dynamic_paths)
        if self.isTrain is True:
            dynamic = self.load_dynamic(dynamic_paths[0])
        
            return_list = {'Image': Images, 'Semantic': Semantics, 'Instance': Instancs, 'Dynamic': dynamic, \
                       'Image_path': image_paths, 'Semantic_path': semantic_paths}
        else:
            return_list = {'Image': Images, 'Semantic': Semantics, 'Instance': Instancs, \
                       'Image_path': image_paths, 'Semantic_path': semantic_paths}
        return return_list

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

    def load_dynamic(self, path):
        img = Image.open(path)
        if img.size[1] != self.height:
            img = img.resize((self.width, self.height), resample=Image.NEAREST)
        img = np.array(img)
        mask = img == 151
        mask = mask.astype(np.int32)
        return torch.from_numpy(mask)


    def numpy2tensor(self, arr, transform_scaleA, is_label=False):
        A_scaled = transform_scaleA(arr)
        if is_label:
            A_scaled *= 255
        return A_scaled

    def get_image_numpy(self, path):
        img = Image.open(path)
        if img.size[1] != self.height:
            img = img.resize((self.width, self.height), resample=Image.NEAREST)
        return img

    def get_image_test(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)        
        if A_img.size[1] != self.height:
            A_img = A_img.resize((self.width, self.height))
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def __len__(self):
        return self.n_of_seqs

    def name(self):
        return 'TemporalDataset'

    def frame2short(self, frame):
        sp = frame.split('_')
        short = sp[0] + '_' + '%d'%int(sp[1]) + '_' + '%d'%int(sp[2]) + '.png'
        return short

    def load_all_image_paths(self, image_dir, semantic_dir, instance_dir, dynamic_dir):
        #Subdir
        city_dir = os.listdir(image_dir)
        city_dir.sort()
        video = []
        for i in range(len(city_dir)):
            frame_dir = image_dir + city_dir[i]
            frame_list = os.listdir(frame_dir)
            frame_list.sort()
            for j in range(len(frame_list)//30):
                image = []
                semantic = []
                instance = []
                dynamic = []
                flag = True
                for k in range(j*30 + 0, (j+1)*30):
                    full_image_path = frame_dir + "/" + frame_list[k]
                    full_semantic_path = semantic_dir + city_dir[i] + "/" + frame_list[k]
                    full_instance_path = instance_dir + city_dir[i] + "/" + frame_list[k]
                    assert os.path.isfile(full_image_path)
                    assert os.path.isfile(full_semantic_path)
                    image.append(full_image_path)
                    semantic.append(full_semantic_path)
                    instance.append(full_instance_path)
                    if k == j*30 + 19 and self.isTrain is True:
                        full_dynamic_path = dynamic_dir + city_dir[i] + '/' + self.frame2short(frame_list[k])
                        #print(full_dynamic_path)
                        assert os.path.isfile(full_dynamic_path)
                        dynamic.append(full_dynamic_path)
                if flag is True:
                    video.append((image, semantic, instance, dynamic))
        return video
    

