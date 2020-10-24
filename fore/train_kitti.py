### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer


# Predict future 5/10 frames at 256x512 / 512 x 1024 resolution




def process_conf(back_mask):
    # Set foreground region as -1 as invalid
    _, _, _, h, w = back_mask
    mask = np.zeros((1,1,1,h,w))
    mask[back_mask == 1] = 1
    mask[back_mask == 0] = -1
    return mask

def compute_flow(flow, tIn, Semantic_input):
    input_flow = [None]*(tIn - 1)
    input_conf = [None]*(tIn - 1)
    for i in range(tIn - 1):
        background_mask = 1.0 - Semantic_input[:,i:i+1,-1:,:,:]
        curr_flow = flow[:,i:i+1,...]
        out_conf = process_conf(out_conf, background_mask)
        input_flow[i], input_conf[i] = out_flow, out_conf
    input_flow_cat = torch.cat([input_flow[k] for k in range(tIn - 1)], dim=1)
    return input_flow_cat

def prepare_input(tensor_list):
    ret = []
    for i in tensor_list:
        ret.append(i.float().cuda())
    return ret


def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1    
        opt.nThreads = 1
        opt.ImagesRoot = '/data/shared/dataset/cityscapes/leftImg8bit_sequence_512p/'
        opt.BackRoot = "/data/shared/dataset/cityscapes/leftImg8bit_sequence_256p_background_inpainted/"
        opt.SemanticRoot = '/data/yue/cityscapes/leftImg8bit_sequence_semantic_256p/'
        opt.Instance_maskrcnn = '/data/shared/dataset/cityscapes/InstanceMap_512p/'
        opt.InstanceGTRoot = "/data/shared/dataset/cityscapes/gtFine/"
        opt.SemanticGTRoot = "/data/shared/dataset/cityscapes/gtFine/"
        opt.niter = 100
        opt.niter_decay = 100

    ### initialize dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training Cityscapes videos = %d' % dataset_size)

    ### initialize models
    modelG, modelD = create_model(opt)
    optimizer_G = modelG.module.optimizer_G
    optimizer_D = modelD.module.optimizer_D
    optimizer_D_T = modelD.module.optimizer_D_T
    visualizer = Visualizer(opt)

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    ### if continue training, recover previous states
    if opt.continue_train:        
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
        if start_epoch > opt.niter:
            modelG.module.update_learning_rate(start_epoch-1)
            modelD.module.update_learning_rate(start_epoch-1)
        if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (start_epoch > opt.niter_fix_global):
            modelG.module.update_fixed_params()
        if start_epoch > opt.niter_step:
            data_loader.dataset.update_training_batch((start_epoch-1)//opt.niter_step)
            modelG.module.update_training_batch((start_epoch-1)//opt.niter_step)
    else:    
        start_epoch, epoch_iter = 1, 0

    ### set parameters    
    n_gpus = len(opt.gpu_ids)            # number of gpus used for generator for each batch
    tIn, tOut = opt.tIn, opt.tOut
    channel_all = opt.semantic_nc + opt.image_nc + opt.flow_nc
    t_scales = opt.n_scales_temporal
    semantic_nc = 1
    flow_nc = opt.flow_nc
    image_nc = 3
    back_image_nc = 3

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    total_steps = total_steps // opt.print_freq * opt.print_freq  

    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for idx, data in enumerate(dataset, start=epoch_iter):
            if len(data) == 1:
                #print(data)
                continue
            #print("idx = ", idx)
            if total_steps % opt.print_freq == 0:
                iter_start_time = time.time()

            epoch_iter += opt.batchSize
            _, n_frames_total, height, width = data['Image'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1
            
            ### Input here
            input_semantic = Variable(data['Semantic'][:, :tIn*semantic_nc, ...]).view(-1, tIn, semantic_nc, height, width)
            input_combine = Variable(data['Combine'][:, :tIn*image_nc, ...]).view(-1, tIn, image_nc, height, width)
            input_flow = compute_flow ....
            target_back_map = Variable(data['Backs'][:, :tIn*image_nc, ...]).view(-1, tIn, image_nc, height, width)
            input_mask = Variable(data['Mask'][:, :tIn*1, ...]).view(-1, tIn, 1, height, width)
            last_object = Variable(data['LastObject']).view(-1, 3, height, width)

            ### Label for loss here
            label_image = Variable(data['Image'][:, tIn*image_nc:(tIn+tOut)*image_nc, ...]).view(-1, tOut, image_nc, height, width)
            target_mask = Variable(data['Back'][:, (tIn)*image_nc:(i+tIn+tOut)*image_nc, ...]).view(-1, tOut, image_nc, height, width)

            warp_rgb, params, warp_mask = modelG(input_combine, input_semantic, input_flow, target_back_map, input_mask, last_object)
            losses, losses_T, pred_image_list, gt_list = modelD([curr_instance, input_back_image, warp_rgb, params, warp_mask, label_image])

            losses = [torch.mean(x) if x is not None else 0 for x in losses]
            losses_T = [torch.mean(x) if x is not None else 0 for x in losses_T]
            loss_dict = dict(zip(modelD.module.loss_names, losses))
            loss_dict_T = dict(zip(modelD.module.loss_names_T, losses_T))

            # collect losses
            loss_G, loss_D, loss_D_T = modelD.module.get_losses(loss_dict, loss_dict_T)
            # update generator weights
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # update discriminator weights
            # individual frame discriminator
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            optimizer_D_T.zero_grad()
            loss_D_T.backward()
            optimizer_D_T.step()

            ### display output images
            if save_fake:
                # print("pred_semantic", pred_semantic[0].size())
                visual_list = []
                for k in range(tOut):
                    visual_list.append(('pred_im_%02d' % k, util.tensor2im(pred_image_list[k][0, ...])))
                    visual_list.append(('real_im_%02d' % k, util.tensor2im(gt_list[k][0, ...])))
                    visual_list.append(('label_image_%02d' % k, util.tensor2im(label_image[0, k, ...])))
                    visual_list.append(('Back_image_%02d' % k, util.tensor2im(input_back_image[0, k, ...])))
                    # visual_list.append(('pred_instance_%02d'%k, util.tensor2label(pred_instance[1][0, :, k*instance_nc:(k+1)*instance_nc, :, :], opt.instance_nc)))
                    for k in range(tIn):
                        visual_list.append(('input_image_%02d' % k, util.tensor2im(input_image[0, k, ...])))
                        visual_list.append(('input_semantic_%02d' % k,
                                                util.tensor2label(input_semantic[0, k, ...], opt.semantic_nc)))
                    visual_list.append(("key_instance", util.tensor2edge(curr_instance[0, ...])))
                    visuals = OrderedDict(visual_list)
                    visualizer.display_current_results(visuals, epoch, total_steps)


            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)



            ### save latest model
            if total_steps % opt.save_latest_freq == 0:
                visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                modelG.module.save('latest')
                modelD.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
        #print("debug")
        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            modelG.module.save('latest')
            modelD.module.save('latest')
            modelG.module.save(epoch)
            modelD.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch % opt.niter == 0:
            modelG.module.update_learning_rate(epoch)
            modelD.module.update_learning_rate(epoch)

        ### gradually grow training sequence length
        #if (epoch % opt.niter_step) == 0:
        #    data_loader.dataset.update_training_batch(epoch//opt.niter_step)
        #    modelG.module.update_training_batch(epoch//opt.niter_step)

        ### finetune all scales
        if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            modelG.module.update_fixed_params()

if __name__ == "__main__":
   train()
