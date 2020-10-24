import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform
import numpy as np
from torch.autograd import Variable
from PIL import Image
import psutil
import cv2

class TemporalDataset(BaseDataset):
    def initialize(self, opt):
        assert opt.dataset == 'cityscapes'
        self.opt = opt
        self.height = int(opt.loadSize/2.0)
        self.width = opt.loadSize
        self.isTrain = opt.isTrain
        #if static, use pre-computed static map
        #if not, use background mask
        self.static = opt.static
        if opt.isTrain == True:
            phase = 'train'
        else:
            phase = 'val'
        self.all_image_paths = self.load_all_image_paths(opt.ImagesRoot + phase + '/', \
                                               opt.SemanticRoot + phase + '/', \
                                               opt.InstanceRoot + phase + '/', \
                                               opt.StaticMapDir + phase + '/')
        self.n_of_seqs = len(self.all_image_paths)                 # number of sequences to train
        print("Load number of video paths = %d"%self.n_of_seqs)
        self.seq_len_max = max([len(A[0]) for A in self.all_image_paths])
        self.n_frames_total = 27


    def __getitem__(self, index):
        tIn = self.opt.tIn
        tOut = self.opt.tOut
        image_paths = self.all_image_paths[index % self.n_of_seqs][0]
        semantic_paths = self.all_image_paths[index % self.n_of_seqs][1]
        instance_paths = self.all_image_paths[index % self.n_of_seqs][2]
        if self.static is True:
            static_paths = self.all_image_paths[index % self.n_of_seqs][3]
            nonrigid_paths = self.all_image_paths[index % self.n_of_seqs][4]
            small_paths =  self.all_image_paths[index % self.n_of_seqs][5]
        # setting parameters
        if self.isTrain == True:
            tAll = tIn + tOut
            start_idx = np.random.randint(0, self.n_frames_total - tAll + 1)
        else:
            tAll = tIn
            start_idx = 0
        # setting transformers
        origin_w, origin_h = Image.open(image_paths[start_idx]).size
        params = get_img_params(self.opt, (origin_w, origin_h))
        transform_scale_bicubic = get_transform(self.opt, params)
        transform_scale_nearest = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        # read in images
        Images = 0
        Semantics = 0
        Instancs = 0
        Back_mask = 0
        for i in range(tAll):
            image_path = image_paths[start_idx + i]
            semantic_path = semantic_paths[start_idx + i]
            instance_path = instance_paths[start_idx + i]
            nonrigid_path = nonrigid_paths[start_idx + i]
            small_path = small_paths[start_idx + i]
            semantic_PIL  = self.get_resize_PIL(semantic_path, Image.NEAREST)
            Semantici = self.numpy2tensor(semantic_PIL, transform_scale_nearest, True)
            image_PIL = self.get_resize_PIL(image_path)
            if self.static:
                non_rigid = self.read_non_rigid(nonrigid_path)
                small = self.read_non_rigid(small_path)
                # Load static
                static_path = static_paths[start_idx + i]
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
                       'Image_path': image_paths, 'Semantic_path': semantic_paths}
        return return_list

    def numpy2tensor(self, arr, transform_scaleA, is_label=False):
        A_scaled = transform_scaleA(arr)
        if is_label:
            A_scaled *= 255
        return A_scaled

    def get_resize_PIL(self, path, method=Image.BICUBIC):
        img = Image.open(path)

        if img.size[1] != self.height:
            img = img.resize((self.width, self.height), resample=method)
        return img

    def delete_non_rigid(self, static_pil, non_rigid, small):
        static = 1.0 - np.array(static_pil)/255.0
        kernel = np.ones((7,7),np.int32)
        non_rigid = cv2.dilate(non_rigid, kernel, iterations=1) 
        static[non_rigid == 1] = 0
        static[small == 1] = 1
        #kernel = np.ones((7,7),np.int32)
        #static = cv2.erode(static, kernel, iterations=1)
        return static

    def get_image_back(self, image_pil, transform_scaleA, backmask):
        A_img = np.array(image_pil)*np.tile(np.expand_dims(backmask, axis=2), [1,1,3])
        A_img = Image.fromarray(A_img.astype(np.uint8))
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def compute_non_rigid(self, semantic_PIL):
        non_rigid = np.zeros((self.height, self.width))   
        non_rigid_idx = [11, 12, 17, 18]
        semantic_npy = np.array(semantic_PIL)
        for b in range(len(non_rigid_idx)):
            non_rigid[semantic_npy == non_rigid_idx[b]] = 1
        return non_rigid

    def read_non_rigid(self, non_rigid_path):
        mask = np.array(Image.open(non_rigid_path).resize((self.width, self.height), resample=Image.NEAREST))/255
        return mask

    def compute_back(self, arr):
        semantic = np.array(arr)
        back = semantic < 11
        return back.astype(np.int32)

    def mask2tensor(self, mask, transform_scaleA):
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        mask_tensor = transform_scaleA(mask)
        mask_tensor *= 255
        #print(mask_tensor.size())
        return  mask_tensor

    def __len__(self):
        return self.n_of_seqs

    def name(self):
        return 'TemporalDataset'

    def load_all_image_paths(self, image_dir, semantic_dir, instance_dir, static_map_dir):
        non_rigid_dir = self.opt.non_rigid_dir
        small_object_mask_dir = self.opt.small_object_mask_dir
        city_dir = os.listdir(image_dir)
        city_dir.sort()
        video = []
        video_cnt = 0
        for i in range(len(city_dir)):
            frame_dir = image_dir + city_dir[i]
            frame_list = os.listdir(frame_dir)
            frame_list.sort()
            for j in range(len(frame_list)//30):
                image = []
                semantic = []
                instance = []
                static_paths = []
                non_rigid_paths = []
                small_object_mask = []
                for k in range(j*30, j*30 + 4):
                    full_image_path = frame_dir + "/" + frame_list[k]
                    full_semantic_path = semantic_dir + city_dir[i] + "/" + frame_list[k]
                    full_instance_path = instance_dir + city_dir[i] + "/" + frame_list[k]
                    
                    #print("inst", full_instance_path)
                    assert os.path.isfile(full_image_path)
                    assert os.path.isfile(full_semantic_path)
                    #assert os.path.isfile(full_instance_path)
                    image.append(full_image_path)
                    semantic.append(full_semantic_path)
                    instance.append(full_instance_path)
                    if self.static is True:
                        #full_static_path = static_map_dir + city_dir[i] + "/" + frame_list[k]
                        full_static_path = static_map_dir + "%04d/"%video_cnt + "pred_dynamic_%02d.png"%(k-j*30)
                        #print(full_static_path)
                        assert os.path.isfile(full_static_path)
                        static_paths.append(full_static_path)

                        full_nonrigid_path = non_rigid_dir + "%04d/"%video_cnt + "non_rigid_mask_%02d.png"%(k-j*30)
                        #print(full_nonrigid_path)
                        assert os.path.isfile(full_nonrigid_path)
                        non_rigid_paths.append(full_nonrigid_path)

                        full_small_object_path = small_object_mask_dir + "%04d/"%video_cnt + "small_object_mask_%02d.png"%(k-j*30)
                        assert os.path.isfile(full_small_object_path)
                        small_object_mask.append(full_small_object_path)

                if self.static is True:
                    video.append((image, semantic, instance, static_paths, non_rigid_paths, small_object_mask))
                    video_cnt = video_cnt + 1
                else:
                    video.append((image, semantic, instance))
        return video
