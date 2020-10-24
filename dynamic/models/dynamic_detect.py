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

class DynamicDetect(BaseModel):
    def name(self):
        return 'DynamicDetect'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.bs = opt.batchSize
        self.tIn = opt.tIn
        self.tOut = opt.tOut
        self.loadSize = opt.loadSize
        if not opt.debug:
            torch.backends.cudnn.benchmark = True       

        semantic_nc = opt.semantic_nc
        image_nc = opt.image_nc
        flow_nc = opt.flow_nc
        conf_nc = 1
        edge_nc = 1
        netG_input_nc = (semantic_nc + image_nc + edge_nc) * opt.tIn + (flow_nc + conf_nc)  * (opt.tIn - 1)
        # output nc: predict semantic/image/flow/mask next frame
        netG_output_nc = (semantic_nc + image_nc) * opt.tOut

        self.netG0 = networks.define_G(netG_input_nc, netG_output_nc, opt.tOut, opt.ngf, opt.netG,
                                       opt.n_downsample_G, opt.norm, 0, self.isTrain, self.gpu_ids, opt)

        print('---------- Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:                    
            self.load_network(getattr(self, 'netG0'), 'G0', opt.which_epoch, opt.load_pretrain)

        # set loss functions and optimizers
        if self.isTrain:            
            self.old_lr = opt.lr
            # initialize optimizer G
            params = list(getattr(self, 'netG0').parameters())
            beta1, beta2 = opt.beta1, 0.999
            lr = opt.lr            
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))


    def encode_input(self, in_semantic, in_inst):        
        size = in_semantic.size()
        self.bs, tIn, self.height, self.width = size[0], size[1], size[3], size[4]
               
        oneHot_size = (self.bs, tIn, 19, self.height, self.width)
        input_s = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_s = input_s.scatter_(2, in_semantic.long(), 1.0)    
        input_s = Variable(input_s)       
        in_edge = Variable(self.get_edges(in_inst))         
        return input_s, in_edge

    def forward(self, input_image, input_semantic, input_flow, input_conf, input_instance):  
        gpu_id = input_image.get_device()  
        input_semantic, input_edge = self.encode_input(input_semantic, input_instance)

        _, _, _ , h, w = input_semantic.size()
        semantic_reshaped = input_semantic.view(self.bs, -1, h, w).cuda(gpu_id)
        # 2. input images
        image_reshaped = input_image.view(self.bs, -1, h, w).cuda(gpu_id)
        # 3. flow inputs
        flow_reshaped = input_flow.view(self.bs, -1, h, w).cuda(gpu_id)
        # 4. conf inputs
        conf_reshaped = input_conf.view(self.bs, -1, h, w).cuda(gpu_id)

        edge_reshaped = input_edge.view(self.bs, -1, h, w).cuda(gpu_id)

        pred_mask = self.netG0.forward(self.loadSize, image_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, edge_reshaped)
        return pred_mask


    def inference(self, input_image, input_semantic, input_flow, input_conf, input_instance):
        # Leave instance map uncomplished  
        with torch.no_grad():
            gpu_id = input_image.get_device()     
            input_semantic, input_edge = self.encode_input(input_semantic, input_instance)

            _, _, _ , h, w = input_semantic.size()
            semantic_reshaped = input_semantic.view(self.bs, -1, h, w).cuda(gpu_id)
            # 2. input images
            image_reshaped = input_image.view(self.bs, -1, h, w).cuda(gpu_id)
            # 3. flow inputs
            flow_reshaped = input_flow.view(self.bs, -1, h, w).cuda(gpu_id)
            # 4. conf inputs
            conf_reshaped = input_conf.view(self.bs, -1, h, w).cuda(gpu_id)

            edge_reshaped = input_edge.view(self.bs, -1, h, w).cuda(gpu_id)
    
            pred_mask \
            = self.netG0.forward(self.loadSize, image_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, edge_reshaped)
            
        return pred_mask

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,:,1:] = edge[:,:,:,:,1:] | (t[:,:,:,:,1:] != t[:,:,:,:,:-1])
        edge[:,:,:,:,:-1] = edge[:,:,:,:,:-1] | (t[:,:,:,:,1:] != t[:,:,:,:,:-1])
        edge[:,:,:,1:,:] = edge[:,:,:,1:,:] | (t[:,:,:,1:,:] != t[:,:,:,:-1,:])
        edge[:,:,:,:-1,:] = edge[:,:,:,:-1,:] | (t[:,:,:,1:,:] != t[:,:,:,:-1,:])
        return edge.float()

    def save(self, label):        
        self.save_network(getattr(self, 'netG0'), 'G0', label, self.gpu_ids)                

    def update_learning_rate(self, epoch):        
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr