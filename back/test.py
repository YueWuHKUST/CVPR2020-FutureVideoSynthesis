import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import sys


def process_conf(conf, mask):
    # Set foreground region as 0 as invalid
    conf[mask == 0] = 0
    return conf

def compute_flow(input_images, tIn, flowNet, backmask_in):
    input_images = input_images.data.cuda()
    input_flow = [None]*(tIn-1)
    input_conf = [None]*(tIn-1)
    for i in range(tIn - 1):
        input_image_a = input_images[:,i,:,:,:]
        input_image_b = input_images[:,i+1,:,:,:]
        background_mask = backmask_in[:,i:i+1,...]
        out_flow, out_conf = flowNet(input_image_a, input_image_b)
        out_flow = out_flow.unsqueeze(1)
        out_conf = out_conf.unsqueeze(1)
        out_conf = process_conf(out_conf, background_mask)
        input_flow[i], input_conf[i] = out_flow, out_conf
    input_flow_cat = torch.cat([input_flow[k] for k in range(tIn - 1)], dim=1)
    input_conf_cat = torch.cat([input_conf[k] for k in range(tIn - 1)], dim=1)
    return input_flow_cat, input_conf_cat

def test():
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    ### set parameters
    n_gpus = 1  # opt.n_gpus_gen // opt.batchSize             # number of gpus used for generator for each batch
    tIn, tOut = opt.tIn, opt.tOut
    channel_all = opt.semantic_nc + opt.image_nc + opt.flow_nc
    semantic_nc = 1
    flow_nc = opt.flow_nc
    image_nc = opt.image_nc
    instance_nc = 1
    opt.dataset_mode = 'test'
    opt.static = True
    
    # Dataset city or vkitti
    modelG, flowNet = create_model(opt)

    data_loader = CreateDataLoader(opt, flowNet)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    visualizer = Visualizer(opt)
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    save_dir = "./result/%s/"%opt.dataset
    print("save_dir = ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Doing %d frames' % len(dataset))
    global_input_image = None
    global_input_semantic = None

    for i, data in enumerate(dataset):
        if i > opt.how_many:
            break
        global_input_image = []
        global_input_semantic = []
        global_input_instance = []
        global_input_back_mask = []
        for pred_id in range(2):
            print("Testing %04d %02d"%(i, pred_id))
            _, _, height, width = data['Image'].size()
            flag = True
            t_all = tIn + tOut
            t_len = t_all

            if pred_id == 0:
                # 5D tensor: batchSize, # of frames, # of channels, height, width
                input_image = Variable(data['Image'][:, 0:tIn*image_nc, ...]).view(-1, tIn, image_nc, height, width)
                input_semantic = Variable(data['Semantic'][:, 0:tIn*semantic_nc, ...]).view(-1, tIn, semantic_nc, height, width)
                input_back_mask = Variable(data['back_mask'][:, 0:tIn,...]).view(-1, tIn, 1, height, width)
                input_instance = Variable(data['Instance'][:, 0:tIn,...]).view(-1, tIn, 1, height, width)  
                global_input_image = input_image
                global_input_semantic = input_semantic
                global_input_instance = input_instance
                global_input_back_mask = input_back_mask
                input_semantic = input_semantic.float().cuda()
                input_back_mask = input_back_mask.float().cuda()
                input_image = input_image.float().cuda()
                input_instance = input_instance.float().cuda()
                if torch.sum((input_back_mask[:,-1,...]==0).float()) > 0:
                    flag = False
                input_flow, input_conf = compute_flow(input_image, tIn, flowNet, input_back_mask)
                input_semantic = input_semantic.float().cuda()
                input_image = input_image.float().cuda()
                #print("input_image",input_image.size(),input_semantic.size(),input_back_mask.size(),input_edge.size(),input_flow.size(),input_conf.size())
                real_image_prev = input_image[:,-1,:,:,:]
                real_semantic_prev = input_semantic[:,-1:,:,:,:]
                real_mask_prev = input_back_mask[:,-1,...]
                pred_flow, warp_image_bwd, warp_mask_bwd, warp_semantic_bwd, warp_instance_bwd = \
                        modelG.inference(input_image, input_semantic, input_flow, input_conf, input_instance, input_back_mask)
                #if flag is True:
                process_image = warp_image_bwd.view(-1, tOut, image_nc, height, width)
                global_input_image = torch.cat((global_input_image, process_image.cpu()), dim=1)
                process_semantic = warp_semantic_bwd.view(-1, tOut, semantic_nc, height, width)
                global_input_semantic = torch.cat((global_input_semantic, process_semantic.cpu()), dim=1)
                process_instance = warp_instance_bwd.view(-1, tOut, 1, height, width)
                global_input_instance = torch.cat((global_input_instance, process_instance.cpu()),dim=1)
                process_mask = warp_mask_bwd.view(-1, tOut, 1, height, width)
                global_input_back_mask = torch.cat((global_input_back_mask, process_mask.cpu()),dim=1)
                visual_list = []
                save_dir_cnt = save_dir + "/%04d/"%i
                print("save_dir_cnt = ", save_dir_cnt)
                if not os.path.exists(save_dir_cnt):
                    os.makedirs(save_dir_cnt)
                for p in range(tIn):
                    visual_list.append(('input_image_%02d'%p, util.tensor2im(input_image[0, p:p+1, :, :, :])))
                    visual_list.append(('input_mask_%02d'%p, util.tensor2mask(input_back_mask[0, p:p+1, :, :, :])))
                for k in range(tOut):
                    visual_list.append(('warp_image_bwd_%02d'%k, util.tensor2im(warp_image_bwd[0, k*3:(k+1)*3, :, :])))
                    visual_list.append(('warp_mask_bwd_%02d'%k, util.tensor2mask(warp_mask_bwd[0, k*1:(k+1)*1, :, :])))
                    visual_list.append(('warp_semantic_bwd_%02d'%k, util.tensor2graysemantic(warp_semantic_bwd[0, k*1:(k+1)*1, ...])))
                    visual_list.append(('warp_instance_bwd_%02d'%k, util.tensor2im(warp_instance_bwd[0, k*1:(k+1)*1, ...])))
                visuals = OrderedDict(visual_list)
                visualizer.save_test_images(save_dir_cnt, visuals, i)
            else:
                if flag is True:
                    input_image = global_input_image[:, -tIn:, :, :, :]
                    input_semantic = global_input_semantic[:, -tIn:, :, :, :]
                    input_instance = global_input_instance[:, -tIn:, :, :, :]
                    input_back_mask = global_input_back_mask[:, -tIn:, :, :, :]
                    input_semantic = input_semantic.float().cuda()
                    input_back_mask = input_back_mask.float().cuda()
                    input_image = input_image.float().cuda()
                    input_instance = input_instance.float().cuda()
                    input_flow, input_conf = compute_flow(input_image, tIn, flowNet, input_back_mask)
                    pred_flow, warp_image_bwd, warp_mask_bwd, warp_semantic_bwd, warp_instance_bwd = \
                        modelG.inference(input_image, input_semantic, input_flow, input_conf, input_instance, input_back_mask)
                    visual_list = []
                    save_dir_cnt = save_dir + "/%04d/"%i
                    print("save_dir_cnt = ", save_dir_cnt)
                    if not os.path.exists(save_dir_cnt):
                        os.makedirs(save_dir_cnt)
                    for k in range(tOut):
                        visual_list.append(('warp_image_bwd_%02d'%(k+tOut), util.tensor2im(warp_image_bwd[0, k*3:(k+1)*3, :, :])))
                        visual_list.append(('warp_mask_bwd_%02d'%(k+tOut), util.tensor2mask(warp_mask_bwd[0, k*1:(k+1)*1, :, :])))
                        visual_list.append(('warp_semantic_bwd_%02d'%(k+tOut), util.tensor2graysemantic(warp_semantic_bwd[0, k*1:(k+1)*1, ...])))
                        visual_list.append(('warp_instance_bwd_%02d'%(k+tOut), util.tensor2im(warp_instance_bwd[0, k*1:(k+1)*1, ...])))
                      
                    visuals = OrderedDict(visual_list)
                    visualizer.save_test_images(save_dir_cnt, visuals, i)
            
            
            



if __name__ == "__main__":
   test()
