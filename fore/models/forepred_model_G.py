### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import math
import torch
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks

# Generator

class ForePredModelG(BaseModel):
    def name(self):
        return 'ForePredModelG'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.bs = opt.batchSize
        self.tIn = opt.tIn
        self.tOut = opt.tOut
        self.loadSize = opt.loadSize
        if not opt.debug:
            torch.backends.cudnn.benchmark = True       
                              
        self.split_gpus = (self.opt.n_gpus_gen < len(self.opt.gpu_ids)) and (self.opt.batchSize == 1)

        semantic_nc = opt.semantic_nc
        image_nc = opt.image_nc
        flow_nc = opt.flow_nc
        conf_nc = 1
        carmask_nc = 1
        netG_input_nc = (semantic_nc + carmask_nc) * opt.tIn + (flow_nc + conf_nc)  * (opt.tIn - 1) + image_nc * opt.tIn + image_nc * opt.tOut
        self.netG0 = networks.define_G(netG_input_nc, opt.dataset, self.opt.loadSize, self.tOut, opt.ngf, self.gpu_ids)
        print('---------- Flow Generation Networks initialized -------------') 


        if not self.isTrain or opt.continue_train or opt.load_pretrain:                    
            self.load_network(getattr(self, 'netG0'), 'G0', opt.which_epoch, opt.load_pretrain)

        if self.isTrain:            
            self.old_lr = opt.lr
            params = list(getattr(self, 'netG0').parameters())
            print('------------ Traning at small resolutions -----------')
            
            beta1, beta2 = 0.9, 0.999
            lr = opt.lr            
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))


    def encode_input(self, dataset, in_semantic):        
        size = in_semantic.size()
        self.bs, tIn, self.height, self.width = size[0], size[1], size[3], size[4]
               
        oneHot_size = (self.bs, tIn, self.opt.semantic_nc, self.height, self.width)
        input_s = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_s = input_s.scatter_(2, in_semantic.long(), 1.0)    
        input_s = Variable(input_s)  
    
        return input_s

    def forward(self, input_combine, input_semantic, input_flow, input_conf, target_back_map, input_mask, last_object):
        tIn = self.opt.tIn
        tOut = self.opt.tOut
        dataset = self.opt.dataset
        gpu_split_id = self.opt.n_gpus_gen + 1        

        gpu_id = input_combine.get_device()
        # broadcast netG to all GPUs used for generator
        netG_0 = getattr(self, 'netG0')                        

        input_semantic = self.encode_input(dataset, input_semantic)

        _, _, _ , h, w = input_semantic.size()
        # Combine
        combine_reshaped = input_combine.view(self.bs, -1, h, w).cuda(gpu_id)
        # Semantic
        semantic_reshaped = input_semantic.view(self.bs, -1, h, w).cuda(gpu_id)
        # flow inputs
        flow_reshaped = input_flow.view(self.bs, -1, h, w).cuda(gpu_id)
        # conf inputs
        conf_reshaped = input_conf.view(self.bs, -1, h, w).cuda(gpu_id)
        # Binary mask
        mask_reshaped = input_mask.view(self.bs, -1 , h, w).cuda(gpu_id)
        # target future inpainted background
        target_back_reshaped = target_back_map.view(self.bs, -1, h, w).cuda(gpu_id)

        warped_object, warped_mask, affine_matrix, pred_complete \
                = netG_0.forward(self.loadSize, combine_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, mask_reshaped, target_back_reshaped)


        return warped_object, warped_mask, affine_matrix, pred_complete


    def inference(self, input_combine, input_semantic, input_flow, input_conf, target_back_map, input_mask, last_object, last_mask):
        # Leave instance map uncomplished
        tIn = self.opt.tIn
        tOut = self.opt.tOut
        dataset = self.opt.dataset

        gpu_id = input_combine.get_device()
        # broadcast netG to all GPUs used for generator
        with torch.no_grad():
            netG_0 = getattr(self, 'netG0')                        
            netG_0 = [netG_0]
            
            input_semantic = self.encode_input(dataset, input_semantic)

            _, _, _ , h, w = input_semantic.size()
            # Combine
            combine_reshaped = input_combine.view(self.bs, -1, h, w).cuda(gpu_id)
            # Semantic
            semantic_reshaped = input_semantic.view(self.bs, -1, h, w).cuda(gpu_id)
            # flow inputs
            flow_reshaped = input_flow.view(self.bs, -1, h, w).cuda(gpu_id)
            # conf inputs
            conf_reshaped = input_conf.view(self.bs, -1, h, w).cuda(gpu_id)
            # Binary mask
            mask_reshaped = input_mask.view(self.bs, -1 , h, w).cuda(gpu_id)
            # target future inpainted background
            target_back_reshaped = target_back_map.view(self.bs, -1, h, w).cuda(gpu_id)

        
            warped_object, warped_mask, affine_matrix, pred_complete \
                = netG_0[0].forward(self.loadSize, combine_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, mask_reshaped, target_back_reshaped, last_object, last_mask)

        return warped_object, warped_mask, affine_matrix, pred_complete

    def save(self, label):        
        self.save_network(getattr(self, 'netG0'), 'G0', label, self.gpu_ids)           

    def update_learning_rate(self, epoch):        
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
