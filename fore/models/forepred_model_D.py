### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

class ForePredModelD(BaseModel):
    # Set two discriminators
    def name(self):
        return 'ForePredModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)        
        gpu_split_id = opt.n_gpus_gen
        self.opt = opt
        self.loadSize = opt.loadSize
        self.tIn = opt.tIn
        self.tOut = opt.tOut
        self.gpu_ids = opt.gpu_ids
        if not opt.debug:
            torch.backends.cudnn.benchmark = True
        ### Initialize Single semantic frame discriminator
        netD_single_input_nc = opt.image_nc + 1
        self.netD = networks.define_D(netD_single_input_nc, opt.ndf, opt.n_layers_D, opt.num_D, gpu_ids=self.gpu_ids)

        ### Initialize Temporal semantic discriminator
        net_D_T_input_nc = 1*(opt.tIn + opt.tOut)
        self.netD_T = networks.define_D(net_D_T_input_nc, opt.ndf, opt.n_layers_D, opt.num_D, gpu_ids=self.gpu_ids)

        print('---------- Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if opt.continue_train or opt.load_pretrain:          
            self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain) 
            print("Loaded Single frame Discriminator")
            self.load_network(self.netD_T, 'D_T', opt.which_epoch, opt.load_pretrain)
            print("Loaded Multi frame Discriminator")      

           
        # set loss functions and optimizers          
        self.old_lr = opt.lr
        self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor)
        self.L1Loss = torch.nn.L1Loss()
        self.L2Loss = torch.nn.MSELoss()
        self.criterionWarp = networks.MaskOneL1Loss()
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'Image', 'Scale', 'Rotation', 'Shear', 'Translation', 'smooth']
        self.loss_names_T = ['G_T_GAN', 'D_T_real', 'D_T_fake']

        params = list(self.netD.parameters())
        beta1, beta2 = opt.beta1, 0.999
        lr = opt.lr
        self.optimizer_D = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
        params_T = list(self.netD_T.parameters())
        self.optimizer_D_T = torch.optim.Adam(params_T, lr=opt.lr, betas=(opt.beta1, 0.999))

    def compute_loss_D(self, netD, pred_image, real_image, fake_mask, real_mask):
        loss_D_real = 0
        loss_D_fake = 0
        loss_G_GAN = 0
        for i in range(self.tOut):
            real_info = torch.cat([real_image[:, i, ...], real_mask[:,i,...]], dim=1)
            fake_info = torch.cat([pred_image[i].detach(), fake_mask[i].detach()], dim=1)
            pred_real = netD.forward(real_info)
            pred_fake = netD.forward(fake_info.detach())
            loss_D_real += self.criterionGAN(pred_real, True)
            loss_D_fake += self.criterionGAN(pred_fake, False)

            pred_fake = netD.forward(fake_info)
            loss_G_GAN += self.GAN_loss(pred_fake)
        return loss_D_real, loss_D_fake, loss_G_GAN

    def compute_loss_D_T(self, real_seq, fake_seq):
        pred_real = self.netD_T.forward(real_seq)
        pred_fake = self.netD_T.forward(fake_seq.detach())
        loss_D_T_real = self.criterionGAN(pred_real, True)
        loss_D_T_fake = self.criterionGAN(pred_fake, False)

        pred_fake = self.netD_T.forward(fake_seq)
        loss_G_T_GAN = self.GAN_loss(pred_fake)

        return loss_D_T_real, loss_D_T_fake, loss_G_T_GAN

    def GAN_loss(self, pred_fake):
        ### GAN loss            
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        return loss_G_GAN

    def gen_seq(self, input_mask, pred_mask, label_mask, tIn, tOut):
        bs, _,  _, height, width = input_mask.size()
        input_mask = input_mask.view(bs, -1, height, width)
        pred_mask_cat = torch.cat([pred_mask[p] for p in range(tOut)], dim=1)
        label_mask = label_mask.view(bs, -1, height, width)
        real_sequence = torch.cat([input_mask.detach(), label_mask], dim=1)
        fake_sequence = torch.cat([input_mask.detach(), pred_mask_cat], dim=1)
        return real_sequence, fake_sequence

    def smooth_loss_affine(self, param):
        # sx sy rotation shear tx ty
        diff_sx = [None]*(self.tOut - 1)
        diff_sy = [None]*(self.tOut - 1)
        diff_rot = [None]*(self.tOut - 1)
        diff_tx = [None]*(self.tOut - 1)
        diff_ty = [None]*(self.tOut - 1)
        diff_shear = [None]*(self.tOut - 1)
        loss = 0
        eps = 1e-4
        for i in range(self.tOut - 1):
            diff_sx[i]   = param[i+1][:, 0] - param[i][:, 0]
            diff_sy[i]   = param[i+1][:, 1] - param[i][:, 1]
            diff_rot[i]  = param[i+1][:, 2] - param[i][:, 2]
            diff_shear[i]= param[i+1][:, 3] - param[i][:, 3]
            diff_tx[i]   = param[i+1][:, 4] - param[i][:, 4]
            diff_ty[i]   = param[i+1][:, 5] - param[i][:, 5]

        for i in range(self.tOut - 2):
            # scale
            diffdiff_sx = diff_sx[i+1] - diff_sx[i]
            loss += self.L1Loss(diffdiff_sx, torch.zeros_like(diffdiff_sx))
            diffdiff_sy = diff_sy[i+1] - diff_sy[i]
            loss += self.L1Loss(diffdiff_sy, torch.zeros_like(diffdiff_sy))
            # rotation
            diffdiff_rot = diff_rot[i+1] - diff_rot[i]
            loss += self.L1Loss(diffdiff_rot, torch.zeros_like(diffdiff_rot))
            # shear
            diffdiff_shear = diff_shear[i+1] - diff_shear[i]
            loss += self.L1Loss(diffdiff_shear, torch.zeros_like(diffdiff_shear))
            # tx
            diffdiff_tx = diff_tx[i+1] - diff_tx[i]
            loss += self.L1Loss(diffdiff_tx, torch.zeros_like(diffdiff_tx))
            # ty
            diffdiff_ty = diff_ty[i+1] - diff_ty[i]
            loss += self.L1Loss(diffdiff_ty, torch.zeros_like(diffdiff_ty))
        return loss


    def forward(self, scale_T, tensors_list):
        scale_S = self.opt.n_scales_spatial
        semantic_nc = self.opt.semantic_nc
        instance_nc = self.opt.instance_nc
        image_nc = self.opt.image_nc
        lambda_D = self.opt.lambda_D
        lambda_D_T = self.opt.lambda_D_T
        lambda_scale = self.opt.lambda_scale
        lambda_shear = self.opt.lambda_shear
        lambda_rotation = self.opt.lambda_rotation
        lambda_translationx = self.opt.lambda_translationx
        lambda_translationy = self.opt.lambda_translationy
        lambda_smooth = self.opt.lambda_smooth
        lambda_image = self.opt.lambda_image
        

        if scale_T > 0:
            # Temporal discriminator loss, input_skipped_frames
            real_seq, fake_seq = tensors_list
            _, _, self.height, self.width = real_seq.size()
            #real_remain = real_remain.view(self.batchSize, -1, self.height, self.width)
            #fake_remain = fake_remain.view(self.batchSize, -1, self.height, self.width)
            if self.loadSize == 1024:
                real_seq = F.interpolate(real_seq, scale_factor=0.5, mode='nearest')
                fake_seq = F.interpolate(fake_seq, scale_factor=0.5, mode='nearest')
            loss_D_T_real, loss_D_T_fake, loss_G_T_GAN = self.compute_loss_D_T(real_seq, fake_seq)
            loss_D_T_real = loss_D_T_real * lambda_D_T
            loss_D_T_fake = loss_D_T_fake * lambda_D_T
            loss_G_T_GAN = loss_G_T_GAN * lambda_D_T
            loss_list = [loss_G_T_GAN, loss_D_T_real, loss_D_T_fake]
            loss_list = [loss.unsqueeze(0) for loss in loss_list]
            return loss_list

        # Construct single frame Discriminart, L1 Loss on prediction frame, and regularization on parameters
        warped_object, warped_mask, affine_matrix, pred_complete, label_combine, target_mask = tensors_list
        self.bs, self.tOut, _ , self.height, self.width = label_combine.size()
        
        # RGB Loss
        rgb_loss = 0
        for i in range(self.tOut):
            rgb_loss += self.criterionWarp(warped_object[i], label_combine[:,i,...], target_mask[:,i,...])
        rgb_loss *= lambda_image
        loss_D_real, loss_D_fake, loss_G_GAN = self.compute_loss_D(self.netD, pred_complete, label_combine, warped_mask, target_mask)
        loss_D_real *= lambda_D
        loss_D_fake *= lambda_D
        loss_G_GAN *= lambda_D


        ################## Regularization on sx and sy
        Scale_Loss = 0
        Rotation_Loss = 0
        Translation_Loss = 0
        Shear_Loss = 0
        Smooth_Loss = 0
        # Smooth_Loss = 0
        # sx sy rotation shear tx ty
        for i in range(self.tOut):
            #print("params =", affine_matrix[i].size())
            sx = affine_matrix[i][:,0]
            sy = affine_matrix[i][:,1]
            Scale_Loss += self.L1Loss(sx, torch.ones_like(sx))
            Scale_Loss += self.L1Loss(sy, torch.ones_like(sy))
            Scale_Loss += self.L1Loss(sx, sy)

            r = affine_matrix[i][:,2]
            s = affine_matrix[i][:,3]
            Rotation_Loss += self.L1Loss(r, torch.zeros_like(r))
            Shear_Loss    += self.L1Loss(s, torch.zeros_like(s))

            tx = affine_matrix[i][:,4]
            ty = affine_matrix[i][:,5]
            Translation_Loss += self.L1Loss(tx, torch.zeros_like(tx))*lambda_translationx
            Translation_Loss += self.L1Loss(ty, torch.zeros_like(ty))*lambda_translationy
        Scale_Loss *= lambda_scale
        Rotation_Loss *= lambda_rotation
        Shear_Loss *= lambda_shear
        #Translation_Loss *= lambda_translation


        Smooth_Loss = self.smooth_loss_affine(affine_matrix)

        Smooth_Loss *= lambda_smooth

        loss_list = [loss_G_GAN, loss_D_real, loss_D_fake, rgb_loss, Scale_Loss, Rotation_Loss, Shear_Loss, Translation_Loss, Smooth_Loss]
        loss_list = [loss.unsqueeze(0) for loss in loss_list]
        return loss_list

    def get_losses(self, loss_dict, loss_dict_T):
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['Image'] + \
            loss_dict['Scale'] + loss_dict['Rotation'] + loss_dict['Shear'] + loss_dict['Translation'] + loss_dict['smooth']

        # collect temporal losses
        loss_G += loss_dict_T['G_T_GAN']
        loss_D_T = (loss_dict_T['D_T_fake'] + loss_dict_T['D_T_real']) * 0.5

        return loss_G, loss_D, loss_D_T


    def save(self, label):
        self.save_network(self.netD, 'D', label, self.gpu_ids)     
        self.save_network(self.netD_T, 'D_T', label, self.gpu_ids)    
       
    def update_learning_rate(self, epoch):        
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

