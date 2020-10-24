import numpy as np
import torch
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel, resample
from . import networks
import torch.nn.functional as F
import copy


class BackPredModelD(BaseModel):
    def name(self):
        return 'BackPredModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)        
        self.opt = opt
        self.tIn = opt.tIn
        self.tOut = opt.tOut
        self.semantic_nc = opt.semantic_nc
        self.image_nc = opt.image_nc
        self.loadSize = opt.loadSize
        self.batchSize = opt.batchSize
        self.dataset = self.opt.dataset
        self.gpu_ids = opt.gpu_ids
        if not opt.debug:
            torch.backends.cudnn.benchmark = True    
        netD_input_nc = opt.image_nc + opt.flow_nc
        print("netD_input_nc", netD_input_nc)
        self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                      opt.num_D, gpu_ids=self.gpu_ids)
        input_len = 1 if self.dataset == 'cityscapes' else 2
        
        netD_input_nc = (opt.image_nc)*(input_len + self.tOut) + (opt.flow_nc)*(self.tOut)
        print("netD_input_nc", netD_input_nc)
        self.netDT1 = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                    opt.num_D, gpu_ids=self.gpu_ids)
       
        print('---------- Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if opt.continue_train or opt.load_pretrain:          
            self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)
            self.load_network(self.netDT1, 'D_T1', opt.which_epoch, opt.load_pretrain)
                
        # set loss functions and optimizers          
        self.old_lr = opt.lr

        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor)
        self.criterionBackMask = networks.MaskOneL1Loss()
        self.criterionWarpImg = networks.MaskOneL1Loss()
        self.criterionWeight = networks.MaskOneL1Loss()
        self.criterionFlow = networks.MaskOneL1Loss()
        self.criterionFeat = torch.nn.L1Loss()
        self.L1loss = torch.nn.L1Loss()
        self.criterionVGG = networks.VGGLoss(self.gpu_ids[0])
        self.smoothness = networks.SmoothLoss()
        self.consist_loss = networks.Consistency()
        self.loss_names = ['G_VGG', 'G_GAN', 'D_real', 'D_fake',
                           'flow', 'Smooth', 'Consistency']
        self.loss_names_T = ['G_T_GAN', 'D_T_real', 'D_T_fake']
        # initialize optimizers D and D_T
        beta1, beta2 = 0.9, 0.999
        lr = opt.lr
        self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()), lr=lr, betas=(beta1, beta2))
        self.optimizer_D_T = torch.optim.Adam(list(self.netDT1.parameters()), lr=lr, betas=(beta1, beta2))

    def compute_loss_D(self, netD, real_image, pred_image, real_flow, pred_flow, backmask_prev, backmask_label):
        # Single image frame discriminator
        loss_D_real = 0
        loss_D_fake = 0
        loss_G_GAN = 0
        loss_G_GAN_Feat = 0
        for i in range(self.tOut):
            cnt_mask_image = backmask_label[:,i:i+1,...].repeat(1,3,1,1)
            cnt_mask_flow = backmask_label[:,i:i+1,...].repeat(1,2,1,1)
            
            cnt_real_image = real_image[:,i*3:(i+1)*3,...]#*cnt_mask_image
            cnt_pred_image = pred_image[:,i*3:(i+1)*3,...]#*cnt_mask_image

            cnt_real_flow = real_flow[:,i*2:(i+1)*2,...]*cnt_mask_flow
            cnt_pred_flow = pred_flow[:,i*2:(i+1)*2,...]*cnt_mask_flow
            cnt_real_cat = torch.cat([cnt_real_image, cnt_real_flow], dim=1)
            cnt_pred_cat = torch.cat([cnt_pred_image, cnt_pred_flow], dim=1)
            #print("cnt_pred_cat", cnt_pred_cat.size())
            pred_real = netD.forward(cnt_real_cat)
            pred_fake = netD.forward(cnt_pred_cat.detach())
            loss_D_real += self.criterionGAN(pred_real, True) 
            loss_D_fake += self.criterionGAN(pred_fake, False)

            pred_fake = netD.forward(cnt_pred_cat)
            loss_G_GAN += self.GAN_loss(pred_fake)

        return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat


    def GAN_loss(self, pred_fake):
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        return loss_G_GAN

    def clip_mask(self, mask):
        one_ = torch.ones_like(mask)
        zero_ = torch.zeros_like(mask)
        return torch.where(mask > 0.5, one_, zero_)

    def compute_loss_D_T(self, netD_T, real_, pred_):
        pred_real = netD_T.forward(real_)
        pred_fake = netD_T.forward(pred_.detach())
        loss_D_T_real = self.criterionGAN(pred_real, True)
        loss_D_T_fake = self.criterionGAN(pred_fake, False)

        pred_fake = netD_T.forward(pred_)
        loss_G_T_GAN = self.GAN_loss(pred_fake)

        return loss_D_T_real, loss_D_T_fake, loss_G_T_GAN

    def get_all_skipped_frames(self, input_image, label_image, pred_image, real_flow, pred_flow, t_scales, backmask_prev, backmask_label):
        bs, _, image_nc, height, width = input_image.size()
        if self.loadSize == 1024:
            pred_image = pred_image.view(bs, -1, height//2, width//2)
            label_image = label_image.view(bs, -1, height, width)
            real_flow = real_flow.view(bs, -1, height, width)
            input_image = input_image.view(bs, -1, height, width)
            #print("input_image size = ", input_image.size())
            backmask_label = backmask_label.view(bs, -1, height, width)

            label_image = F.interpolate(label_image, scale_factor=0.5, mode='bilinear')
            input_image = F.interpolate(input_image, scale_factor=0.5, mode='bilinear')
            real_flow = F.interpolate(real_flow, scale_factor=0.5, mode='nearest')/2.0
            #print("input_image size = ", input_image.size())
            pred_flow = F.interpolate(pred_flow, scale_factor=0.5, mode='nearest')/2.0
            backmask_prev = F.interpolate(backmask_prev, scale_factor=0.5, mode='nearest')
            backmask_label = F.interpolate(backmask_label, scale_factor=0.5, mode='nearest')
        else:
            pred_image = pred_image.view(bs, -1, height, width)
            label_image = label_image.view(bs, -1, height, width)
            real_flow = real_flow.view(bs, -1, height, width)
            input_image = input_image.view(bs, -1, height, width)
            backmask_label = backmask_label.view(bs, -1, height, width)
       
        dataset = self.opt.dataset
        #print(real_flow.size(), pred_flow.size())
        if t_scales > 0:
            real_B_all = get_skipped_frames(input_image, label_image, real_flow, t_scales, backmask_prev, backmask_label, dataset)
            fake_B_all = get_skipped_frames(input_image, pred_image, pred_flow, t_scales, backmask_prev, backmask_label, dataset)
        return real_B_all, fake_B_all

    

    

    def forward(self, scale_T, tensors_list):
        lambda_VGG = self.opt.lambda_VGG
        lambda_F = self.opt.lambda_F
        lambda_smooth = self.opt.lambda_smooth
        lambda_D = self.opt.lambda_D
        lambda_D_T = self.opt.lambda_D_T
        loadSize = self.opt.loadSize
        
        if scale_T > 0:
            # Temporal discriminator loss, input_skipped_frames
            real_remain, fake_remain = tensors_list
            _, _, self.height, self.width = real_remain.size()
            #real_remain = real_remain.view(self.batchSize, -1, self.height, self.width)
            #fake_remain = fake_remain.view(self.batchSize, -1, self.height, self.width)
            if self.loadSize == 1024:
                real_remain = F.interpolate(real_remain, scale_factor=0.5, mode='bilinear')
                fake_remain = F.interpolate(fake_remain, scale_factor=0.5, mode='bilinear')
            loss_D_T_real, loss_D_T_fake, loss_G_T_GAN = self.compute_loss_D_T(self.netDT1, real_remain, fake_remain)
            loss_D_T_real = loss_D_T_real * lambda_D_T
            loss_D_T_fake = loss_D_T_fake * lambda_D_T
            loss_G_T_GAN = loss_G_T_GAN * lambda_D_T
            loss_list = [loss_G_T_GAN, loss_D_T_real, loss_D_T_fake]
            loss_list = [loss.unsqueeze(0) for loss in loss_list]
            return loss_list
            
    

        pred_flow, real_image, real_image_prev, real_flow, real_conf, \
        real_semantic, backmask_prev, backmask_label = tensors_list
        _, _, _, self.height, self.width = real_image.size()
        real_image = real_image.view(-1,self.tOut*3,self.height,self.width)
        #real_semantic = real_semantic.view(-1,self.tOut*1,self.height,self.width)
        backmask_label = backmask_label.view(-1,self.tOut*1,self.height,self.width)    
        real_flow = real_flow.view(-1, self.tOut*4, self.height, self.width)
        real_conf = real_conf.view(-1, self.tOut*2, self.height, self.width)
        
        


        # Downsample to save gpu memory
        
        if self.loadSize == 1024:
            #pred_image = F.interpolate(pred_image, scale_factor=0.5, mode='bilinear')
            for sc in range(3):
                pred_flow[sc] = F.interpolate(pred_flow[sc], scale_factor=0.5, mode='bilinear')/2.0
            real_image = F.interpolate(real_image, scale_factor=0.5, mode='bilinear')
            real_image_prev = F.interpolate(real_image_prev, scale_factor=0.5, mode='bilinear')
            #real_semantic = F.interpolate(real_semantic, scale_factor=0.5, mode='nearest')
            real_flow = F.interpolate(real_flow, scale_factor=0.5, mode='bilinear')/2.0
            backmask_prev = F.interpolate(backmask_prev, scale_factor=0.5, mode='nearest')
            real_conf = F.interpolate(real_conf, scale_factor=0.5, mode='nearest')
            backmask_label = F.interpolate(backmask_label, scale_factor=0.5, mode='nearest')
        
        
        real_flow_fwd = real_flow[:,:self.tOut*2,...]
        real_flow_bwd = real_flow[:,self.tOut*2:,...]
        real_conf_fwd = real_conf[:,:self.tOut,...]
        real_conf_bwd = real_conf[:,self.tOut:,...]
        multi_scale_weight=[1.0, 0.25, 0.0625]

        # Construct weight loss
        #dummy0 = torch.ones_like(img_conf)
        #loss_W = self.criterionFlow(img_conf, dummy0, real_conf_bwd) * lambda_W

        # Multi scale flow loss
        flow_loss = 0#torch.zeros_like(loss_W)
        loss_G_VGG = 0#torch.zeros_like(loss_W)
        loss_smooth = 0#torch.zeros_like(loss_W)
        consistency_loss = 0#torch.zeros_like(loss_W)
        
        bwd_warp_img_list = []
        bwd_warp_mask_list = []
        fwd_warp_img_list = []
        fwd_warp_mask_list = []
        gt_warp_img_list = []
        occlusion_bwd = []
        occlusion_fwd = []
        gpu_id = real_image.get_device()

        for s in range(3):
            cnt_img_prev_s = F.interpolate(real_image_prev, scale_factor=(0.5)**s, mode='bilinear')
            cnt_mask_prev_s = F.interpolate(backmask_prev, scale_factor=(0.5)**s, mode='nearest')

            gt_warp_img_list_s = []
            bwd_warp_img_list_s = []
            bwd_warp_mask_list_s = []
            fwd_warp_img_list_s = []
            fwd_warp_mask_list_s = []
            occlusion_bwd_s = []
            occlusion_fwd_s = []

            for i in range(self.tOut):
                cnt_gt_flow_fwd = real_flow_fwd[:,i*2:(i+1)*2,:,:]
                cnt_gt_flow_bwd = real_flow_bwd[:,i*2:(i+1)*2,...]
                cnt_gt_conf_fwd = real_conf_fwd[:,i*1:(i+1)*1,:,:]
                cnt_gt_conf_bwd = real_conf_bwd[:,i*1:(i+1)*1,:,:]

                cnt_real = real_image[:,i*3:(i+1)*3,:,:]
                cnt_mask = backmask_label[:,i:i+1,...]
                cnt_flow_fwd = pred_flow[s][:,i*2:(i+1)*2,...]
                cnt_flow_bwd = pred_flow[s][:,self.tOut*2+i*2:self.tOut*2+(i+1)*2,...]

                cnt_gt_flow_s_fwd = F.interpolate(cnt_gt_flow_fwd, scale_factor=(0.5)**s, mode='nearest') / (2**s)
                cnt_gt_flow_s_bwd = F.interpolate(cnt_gt_flow_bwd, scale_factor=(0.5)**s, mode='nearest') / (2**s)
                cnt_gt_conf_s_fwd = F.interpolate(cnt_gt_conf_fwd, scale_factor=(0.5)**s, mode='nearest') 
                cnt_gt_conf_s_bwd = F.interpolate(cnt_gt_conf_bwd, scale_factor=(0.5)**s, mode='nearest') 

                cnt_img_label_s = F.interpolate(cnt_real, scale_factor=(0.5)**s, mode='bilinear')
                cnt_mask_label_s = F.interpolate(cnt_mask, scale_factor=(0.5)**s, mode='nearest')

                bwd_cnt_warp_img  = self.resample(cnt_img_prev_s.cuda(gpu_id),  cnt_flow_bwd).cuda(gpu_id)
                #print("bwd_cnt_warp_img", bwd_cnt_warp_img.size())
                bwd_cnt_warp_mask = self.resample(cnt_mask_prev_s.cuda(gpu_id), cnt_flow_bwd).cuda(gpu_id)
                bwd_cnt_warp_mask = self.clip_mask(bwd_cnt_warp_mask)

                src_cnt_warp_img  = self.resample(cnt_img_label_s.cuda(gpu_id), cnt_flow_fwd).cuda(gpu_id)
                src_cnt_warp_mask = self.resample(cnt_mask_label_s.cuda(gpu_id),cnt_flow_fwd).cuda(gpu_id)
                src_cnt_warp_mask = self.clip_mask(src_cnt_warp_mask)

                # After Warp without value don't have loss, because black region mismatch, but can be reasonable
                # Assume the dynamic object in pred and label are the same, dynamic object don't need flow

                cnt_warp_bwd_mask_all = cnt_mask_label_s #* bwd_cnt_warp_mask
                loss_G_VGG += self.criterionVGG(bwd_cnt_warp_img, cnt_img_label_s, cnt_warp_bwd_mask_all) * lambda_VGG * multi_scale_weight[s]

                cnt_warp_fwd_mask_all = cnt_mask_prev_s  #* src_cnt_warp_mask
                loss_G_VGG += self.criterionVGG(src_cnt_warp_img, cnt_img_prev_s,  cnt_warp_fwd_mask_all) * lambda_VGG * multi_scale_weight[s]

                #flow_loss += self.criterionFlow(cnt_flow_bwd, cnt_gt_flow_s_bwd, cnt_warp_bwd_mask_all ) * lambda_F * multi_scale_weight[s]
                flow_loss += self.criterionFlow(cnt_flow_bwd, cnt_gt_flow_s_bwd, cnt_warp_bwd_mask_all * cnt_gt_conf_s_bwd) * lambda_F * multi_scale_weight[s]

                flow_loss += self.criterionFlow(cnt_flow_fwd, cnt_gt_flow_s_fwd, cnt_warp_fwd_mask_all * cnt_gt_conf_s_fwd) * lambda_F * multi_scale_weight[s]
                #flow_loss += self.criterionFlow(cnt_flow_fwd, cnt_gt_flow_s_fwd, cnt_gt_conf_s_fwd) * lambda_F * multi_scale_weight[s]
                
                loss_smooth += self.smoothness(cnt_flow_bwd, cnt_img_label_s, cnt_warp_bwd_mask_all) * lambda_smooth * multi_scale_weight[s]
                loss_smooth += self.smoothness(cnt_flow_fwd, cnt_img_prev_s,  cnt_warp_fwd_mask_all) * lambda_smooth * multi_scale_weight[s]   
                

                fwd_mask, bwd_mask, check_loss = self.consist_loss(cnt_flow_fwd, cnt_flow_bwd, s, cnt_warp_bwd_mask_all, cnt_warp_fwd_mask_all)
                check_loss = check_loss * multi_scale_weight[s]
                consistency_loss += check_loss
                
                gt_warp_img_list_s.append(self.resample(cnt_img_prev_s.cuda(gpu_id), cnt_gt_flow_s_bwd))
                bwd_warp_img_list_s.append(bwd_cnt_warp_img)
                bwd_warp_mask_list_s.append(bwd_cnt_warp_mask)
                fwd_warp_img_list_s.append(src_cnt_warp_img)
                fwd_warp_mask_list_s.append(src_cnt_warp_mask)
                occlusion_bwd_s.append(bwd_mask)
                occlusion_fwd_s.append(fwd_mask)
            gt_warp_img_list_s_cat = torch.cat([gt_warp_img_list_s[p] for p in range(self.tOut)], dim=1)
            bwd_warp_img_list_s_cat = torch.cat([bwd_warp_img_list_s[p] for p in range(self.tOut)], dim=1)
            bwd_warp_mask_list_s_cat = torch.cat([bwd_warp_mask_list_s[p] for p in range(self.tOut)], dim=1)
            fwd_warp_img_list_s_cat = torch.cat([fwd_warp_img_list_s[p] for p in range(self.tOut)], dim=1)
            fwd_warp_mask_list_s_cat = torch.cat([fwd_warp_mask_list_s[p] for p in range(self.tOut)], dim=1)
            occlusion_bwd_s_cat = torch.cat([occlusion_bwd_s[p] for p in range(self.tOut)], dim=1)
            occlusion_fwd_s_cat = torch.cat([occlusion_fwd_s[p] for p in range(self.tOut)], dim=1)
            
            bwd_warp_img_list.append(bwd_warp_img_list_s_cat)
            bwd_warp_mask_list.append(bwd_warp_mask_list_s_cat)
            fwd_warp_img_list.append(fwd_warp_img_list_s_cat)
            fwd_warp_mask_list.append(fwd_warp_mask_list_s_cat)
            gt_warp_img_list.append(gt_warp_img_list_s_cat)
            occlusion_bwd.append(occlusion_bwd_s_cat)
            occlusion_fwd.append(occlusion_fwd_s_cat)


        loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat = self.compute_loss_D(self.netD, real_image, bwd_warp_img_list[0], \
                real_flow_bwd, pred_flow[0][:,self.tOut*2:,...], backmask_prev, backmask_label)
        loss_D_real = loss_D_real * lambda_D
        loss_D_fake = loss_D_fake * lambda_D
        loss_G_GAN = loss_G_GAN * lambda_D
        #print("consistency_loss = ", consistency_loss)
        #print("structure_loss = ", structure_loss)
        loss_list = [loss_G_VGG, loss_G_GAN,
                     loss_D_real, loss_D_fake, flow_loss, loss_smooth, consistency_loss]
        loss_list = [loss.unsqueeze(0) for loss in loss_list]           
        return loss_list, bwd_warp_img_list, bwd_warp_mask_list, occlusion_bwd, occlusion_fwd, gt_warp_img_list


    def get_losses(self, loss_dict, loss_dict_T):                                             
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_VGG'] + \
            loss_dict['Smooth'] + loss_dict['Consistency']
        loss_G += loss_dict['flow']
        # collect temporal losses
        loss_G += loss_dict_T['G_T_GAN']
        loss_D_T = (loss_dict_T['D_T_fake'] + loss_dict_T['D_T_real']) * 0.5
 
        return loss_G, loss_D, loss_D_T
    
    def save(self, label):
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.netDT1, 'D_T1', label, self.gpu_ids)


       
    def update_learning_rate(self, epoch):        
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

# get temporally subsampled frames for real/fake sequences
def get_skipped_frames(input_image, image_out, flow, t_scales, backmask_prev, backmask_label, dataset):
    # image out = B x tOut*3 ...
    bs, _ , height, width = input_image.size()
    tOut = int(image_out.size()[1]/3)
    out_im = [None]*tOut
    out_flow = [None]*tOut
    for i in range(tOut):
        out_im[i] = image_out[:,i*3:(i+1)*3,...]#*backmask_label[:,i,...].repeat(1,3,1,1)
        out_flow[i] = (flow[:, i*2:(i+1)*2,...])*backmask_label[:,i:i+1,...].repeat(1,2,1,1)

    out_i = torch.cat([out_im[p] for p in range(tOut)], dim=1)
    out_f = torch.cat([out_flow[p] for p in range(tOut)], dim=1)
    if dataset == 'cityscapes':
        in_im = input_image[:,-3:,...]
    else:
        in_im = input_image[:,-6:,...]
    remain_all = torch.cat([in_im.detach(), out_i, out_f],dim=1)
    return remain_all
