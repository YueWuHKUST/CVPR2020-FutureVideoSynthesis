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


class BackPredModelG(BaseModel):
    def name(self):
        return 'BackPredModelG'

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
        backmask_nc = 1
        netG_input_nc = (semantic_nc + image_nc + edge_nc + backmask_nc) * opt.tIn + (flow_nc + conf_nc)  * (opt.tIn - 1)
        # output nc: predict semantic/image/flow/mask next frame
        netG_output_nc = (semantic_nc + image_nc) * opt.tOut

        self.netG0 = networks.define_G(netG_input_nc, netG_output_nc, self.tOut, opt.ngf, 
                                       opt.n_downsample_G, opt.norm, 0, self.isTrain, self.gpu_ids, opt)
        print('---------- Flow Generation Networks initialized -------------') 

        if not self.isTrain or opt.continue_train or opt.load_pretrain:                    
            self.load_network(getattr(self, 'netG0'), 'G0', opt.which_epoch, opt.load_pretrain)

        # set loss functions and optimizers
        if self.isTrain:            
            self.old_lr = opt.lr
            params = list(getattr(self, 'netG0').parameters())
            beta1, beta2 = 0.9, 0.999
            lr = opt.lr            
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))


    def encode_input(self, in_semantic, in_inst):        
        size = in_semantic.size()
        self.bs, tIn, self.height, self.width = size[0], size[1], size[3], size[4]
               
        oneHot_size = (self.bs, tIn, self.opt.semantic_nc, self.height, self.width)
        input_s = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_s = input_s.scatter_(2, in_semantic.long(), 1.0)    
        input_s = Variable(input_s)   
        in_edge = Variable(self.get_edges(in_inst))         
        return input_s, in_edge

    def clip_mask(self, mask):
        one_ = torch.ones_like(mask)
        zero_ = torch.zeros_like(mask)
        return torch.where(mask > 0.5, one_, zero_)
    
    def forward(self, input_image, input_semantic_, input_flow, input_conf, input_instance, backmask_in):
        # Leave instance map uncomplished
        tIn = self.opt.tIn
        tOut = self.opt.tOut
        dataset = self.opt.dataset 

        gpu_id = input_image.get_device()
        # broadcast netG to all GPUs used for generator
        netG_0 = getattr(self, 'netG0')                        
        #print(gpu_id)
        input_semantic, input_edge = self.encode_input(input_semantic_, input_instance)

        _, _, _ , h, w = input_semantic.size()
        semantic_reshaped = input_semantic.view(self.bs, -1, h, w).cuda(gpu_id)
        # 2. input images
        image_reshaped = input_image.view(self.bs, -1, h, w).cuda(gpu_id)
        # 3. flow inputs
        flow_reshaped = input_flow.view(self.bs, -1, h, w).cuda(gpu_id)
        # 4. conf inputs
        conf_reshaped = input_conf.view(self.bs, -1, h, w).cuda(gpu_id)

        edge_reshaped = input_edge.view(self.bs, -1, h, w).cuda(gpu_id)
        
        backmask_reshaped = backmask_in.view(self.bs, -1, h, w).cuda(gpu_id)

        predict_flow, up2_conv, input_info \
                = netG_0.forward(self.loadSize, image_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, edge_reshaped, backmask_reshaped)

        return predict_flow, input_edge


    def inference(self, input_image, input_semantic_, input_flow, input_conf, input_instance, backmask_in):
        # Leave instance map uncomplished
        tIn = self.opt.tIn
        tOut = self.opt.tOut
        dataset = self.opt.dataset

        # test for cityscapes
        input_image_origin = input_image
        input_semantic_origin = input_semantic_
        input_flow_origin = input_flow
        input_conf_origin = input_conf
        input_instance_origin = input_instance
        backmask_in_origin = backmask_in

        gpu_id = input_image.get_device()
        # broadcast netG to all GPUs used for generator
        with torch.no_grad():
            netG_0 = getattr(self, 'netG0')                        

            input_semantic, input_edge = self.encode_input(input_semantic_, input_instance)

            _, _, _ , h, w = input_semantic.size()
            semantic_reshaped = input_semantic.view(self.bs, -1, h, w).cuda(gpu_id)
            semantic_reshaped = F.interpolate(semantic_reshaped, scale_factor=0.5, mode='nearest') if self.loadSize == 1024 else semantic_reshaped
            # 2. input images
            image_reshaped = input_image.view(self.bs, -1, h, w).cuda(gpu_id)
            image_reshaped = F.interpolate(image_reshaped, scale_factor=0.5, mode='bilinear') if self.loadSize==1024 else image_reshaped
            # 3. flow inputs
            flow_reshaped = input_flow.view(self.bs, -1, h, w).cuda(gpu_id)
            flow_reshaped = F.interpolate(flow_reshaped, scale_factor=0.5, mode='bilinear')/2.0 if self.loadSize == 1024 else flow_reshaped
            # 4. conf inputs
            conf_reshaped = input_conf.view(self.bs, -1, h, w).cuda(gpu_id)
            conf_reshaped = F.interpolate(conf_reshaped, scale_factor=0.5, mode='nearest') if self.loadSize == 1024 else conf_reshaped

            edge_reshaped = input_edge.view(self.bs, -1, h, w).cuda(gpu_id)
            edge_reshaped = F.interpolate(edge_reshaped, scale_factor=0.5, mode='nearest') if self.loadSize == 1024 else edge_reshaped
        
            backmask_reshaped = backmask_in.view(self.bs, -1, h, w).cuda(gpu_id)
            backmask_reshaped = F.interpolate(backmask_reshaped, scale_factor=0.5, mode='nearest') if self.loadSize == 1024 else backmask_reshaped

            predict_flow, up2_conv, input_info \
                    = netG_0.forward(self.loadSize, image_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, edge_reshaped, backmask_reshaped)


            # Test result at 512x1024 level
            if self.loadSize == 1024:
                predict_flow[0] = F.interpolate(predict_flow[0], scale_factor=2, mode='bilinear') * 2.0


            #### Add warp for testing\
            bwd_warp_img_list = []
            bwd_warp_mask_list = []
            bwd_semantic_list = []
            bwd_instance_list = []
            for i in range(self.tOut):
                cnt_flow_fwd = predict_flow[0][:,i*2:(i+1)*2,...]
                cnt_flow_bwd = predict_flow[0][:,self.tOut*2+i*2:self.tOut*2+(i+1)*2,...]

                bwd_cnt_warp_img  = self.resample(input_image_origin[:,-1,...].cuda(gpu_id),  cnt_flow_bwd).cuda(gpu_id)
                bwd_cnt_warp_mask = self.resample(backmask_in_origin[:,-1,...].cuda(gpu_id), cnt_flow_bwd).cuda(gpu_id)
                bwd_cnt_warp_mask = self.clip_mask(bwd_cnt_warp_mask)
                bwd_cnt_warp_semantic = self.resample(input_semantic_origin[:,-1,...].cuda(gpu_id), cnt_flow_bwd, 'nearest').cuda(gpu_id)
                bwd_cnt_warp_instance = self.resample(input_instance_origin[:,-1,...].cuda(gpu_id), cnt_flow_bwd, 'nearest').cuda(gpu_id)

                bwd_warp_img_list.append(bwd_cnt_warp_img)
                bwd_warp_mask_list.append(bwd_cnt_warp_mask)
                bwd_semantic_list.append(bwd_cnt_warp_semantic)
                bwd_instance_list.append(bwd_cnt_warp_instance)
            bwd_warp_img_list_cat = torch.cat([bwd_warp_img_list[p] for p in range(self.tOut)], dim=1)
            bwd_warp_mask_list_cat = torch.cat([bwd_warp_mask_list[p] for p in range(self.tOut)], dim=1)
            bwd_warp_seamntic_list_cat = torch.cat([bwd_semantic_list[p] for p in range(self.tOut)], dim=1)
            bwd_warp_instance_list_cat = torch.cat([bwd_instance_list[p] for p in range(self.tOut)], dim=1)
        return predict_flow, bwd_warp_img_list_cat, bwd_warp_mask_list_cat, bwd_warp_seamntic_list_cat, bwd_warp_instance_list_cat

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
