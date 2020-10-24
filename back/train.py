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
import copy


'''
Code for background prediction Model



Cityscapes:
    train at 256x512 resolution.
    During test, upsample the predicted flow to obtain 512x1024 results

    Each sequence contains 9 frames, 4(input), 5(output)


Kitti
    Resolution: 256x832

    Each sequence contains 7 frames, 4(input), 3(output)
'''




def process_conf(conf, mask):
    # Set foreground region as 0 as invalid
    conf[mask == 0] = 0
    return conf

def compute_flow(input_images, label_images, tIn, tOut, flowNet, backmask_in, backmask_out):
    input_images = input_images.data.cuda()
    label_images = label_images.data.cuda()
    input_flow = [None]*(tIn-1)
    input_conf = [None]*(tIn-1)
    label_flow_fwd = [None]*tOut
    label_conf_fwd = [None]*tOut
    label_flow_bwd = [None]*tOut
    label_conf_bwd = [None]*tOut

    # Compute forward optical flow between consecutive frames in input frames
    for i in range(tIn - 1):
        input_image_a = input_images[:,i,:,:,:]
        input_image_b = input_images[:,i+1,:,:,:]
        background_mask = backmask_in[:,i:i+1,...]
        # Forward flow
        out_flow, out_conf = flowNet(input_image_a, input_image_b)
        out_flow = out_flow.unsqueeze(1)
        out_conf = out_conf.unsqueeze(1)
        out_conf = process_conf(out_conf, background_mask)
        input_flow[i], input_conf[i] = out_flow, out_conf

    # Compute forward and backward optical flow between last input frame and label frames
    for i in range(tOut):
        input_image_a = input_images[:,-1,...]
        input_image_b = label_images[:,i,...]
        # Forward flow
        background_mask = backmask_in[:,-1:,...]
        out_flow, out_conf = flowNet(input_image_a, input_image_b)
        out_flow = out_flow.unsqueeze(1)
        out_conf = out_conf.unsqueeze(1)
        out_conf = process_conf(out_conf, background_mask)
        label_flow_fwd[i], label_conf_fwd[i] = out_flow, out_conf
        background_mask = backmask_out[:,i:i+1,...]
        # Backward Flow
        out_flow, out_conf = flowNet(input_image_b, input_image_a)
        out_flow = out_flow.unsqueeze(1)
        out_conf = out_conf.unsqueeze(1)
        out_conf = process_conf(out_conf, background_mask)
        label_flow_bwd[i], label_conf_bwd[i] = out_flow, out_conf
    input_flow_cat = torch.cat([input_flow[k] for k in range(tIn - 1)], dim=1)
    input_conf_cat = torch.cat([input_conf[k] for k in range(tIn - 1)], dim=1)
    label_flow_fwd_cat = torch.cat([label_flow_fwd[k] for k in range(tOut)], dim=1)
    label_conf_fwd_cat = torch.cat([label_conf_fwd[k] for k in range(tOut)], dim=1)
    label_flow_bwd_cat = torch.cat([label_flow_bwd[k] for k in range(tOut)], dim=1)
    label_conf_bwd_cat = torch.cat([label_conf_bwd[k] for k in range(tOut)], dim=1)
    label_flow_cat = torch.cat([label_flow_fwd_cat, label_flow_bwd_cat], dim=1)
    label_conf_cat = torch.cat([label_conf_fwd_cat, label_conf_bwd_cat], dim=1)
    return input_flow_cat, input_conf_cat, label_flow_cat, label_conf_cat


def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1    
        opt.nThreads = 1


    ### initialize models
    modelG, modelD, flowNet = create_model(opt)
    visualizer = Visualizer(opt)


    ### initialize dataset
    data_loader = CreateDataLoader(opt, flowNet)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training %s Videos = %d' % (opt.dataset, dataset_size))

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
    else:    
        start_epoch, epoch_iter = 1, 0

    tIn, tOut = opt.tIn, opt.tOut
    image_nc = 3
    semantic_nc = 1
    instance_nc = 1

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    opt.static = True
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    total_steps = total_steps // opt.print_freq * opt.print_freq  

    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        all_loss = np.zeros((7000, 20))
        for idx, data in enumerate(dataset, start=epoch_iter):        
            if total_steps % opt.print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            save_fake = total_steps % opt.display_freq == 0
            _, _, height, width = data['Image'].size()
            t_all = tIn + tOut
            
            input_image = Variable(data['Image'][:, :tIn*image_nc, ...]).view(-1, tIn, image_nc, height, width)
            label_image = Variable(data['Image'][:, tIn*image_nc:(tIn+tOut)*image_nc, ...]).view(-1, tOut, image_nc, height, width)
            input_semantic = Variable(data['Semantic'][:, :tIn*semantic_nc, ...]).view(-1, tIn, semantic_nc, height, width)
            label_semantic = Variable(data['Semantic'][:, tIn*semantic_nc:(tIn+tOut) * semantic_nc, ...]).view(-1,tOut,semantic_nc,height,width)
            input_back_mask = Variable(data['back_mask'][:, :tIn,...]).view(-1, tIn, 1, height, width)
            label_back_mask = Variable(data['back_mask'][:, tIn:tIn+tOut,...]).view(-1, tOut, 1, height, width)
            input_instance = Variable(data['Instance'][:, :tIn,...]).view(-1, tIn, 1, height, width)
            label_instance = Variable(data['Instance'][:, tIn:tIn+tOut,...]).view(-1, tOut, 1, height, width)
            input_instance = input_instance.float().cuda()
            label_instance = label_instance.float().cuda()
            input_semantic = input_semantic.float().cuda()
            label_semantic = label_semantic.float().cuda()
            input_back_mask = input_back_mask.float().cuda()
            label_back_mask = label_back_mask.float().cuda()
            input_image = input_image.float().cuda()
            label_image = label_image.float().cuda()
                
            input_flow, input_conf, label_flow, label_conf = compute_flow(input_image, label_image, tIn, tOut, flowNet, input_back_mask, label_back_mask)
            real_image_prev = input_image[:,-1,:,:,:]
            real_semantic_prev = input_semantic[:,-1:,:,:,:]
            real_mask_prev = input_back_mask[:,-1,...]
            pred_flow, input_edge = \
                modelG(input_image, input_semantic, input_flow, input_conf, input_instance, input_back_mask)
            
            losses, warp_image_bwd, warp_mask_bwd, occlusion_bwd, occlusion_fwd, gt_warp = \
                modelD(0, [pred_flow, label_image, real_image_prev, label_flow, label_conf, label_semantic, \
                real_mask_prev, label_back_mask])
            losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
            loss_dict = dict(zip(modelD.module.loss_names, losses))
            frames_real_remain, frames_fake_remain = \
                modelD.module.get_all_skipped_frames(input_image, label_image, warp_image_bwd[0], \
                        label_flow[:,tOut:,...], pred_flow[0][:,tOut*2:,...], 1, real_mask_prev, label_back_mask)

            # run discriminator for temporal scale
            losses = modelD(1, [frames_real_remain, frames_fake_remain])
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict_T = dict(zip(modelD.module.loss_names_T, losses)) 

            # collect losses
            loss_G, loss_D, loss_D_T = modelD.module.get_losses(loss_dict, loss_dict_T)
           
            ###################################### Backward Pass ################################# 
            optimizer_G = modelG.module.optimizer_G
            optimizer_D = modelD.module.optimizer_D  
            optimizer_D_T = modelD.module.optimizer_D_T

            # update generator weights
            optimizer_G.zero_grad()            
            loss_G.backward()        
            optimizer_G.step()

            # individual frame discriminator
            optimizer_D.zero_grad()
            loss_D.backward()            
            optimizer_D.step()

            # sequence discriminator
            optimizer_D_T.zero_grad()
            loss_D_T.backward()
            optimizer_D_T.step()

            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            errors = {k: v.data.item() for k, v in loss_dict.items()}
            loss_cnt = 0
            for k, v in sorted(errors.items()):
                all_loss[idx, loss_cnt] = v
                loss_cnt += 1

            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors_mean(epoch, epoch_iter, errors, t, all_loss)


            ### display output images
            if save_fake:
                visual_list = []
                for p in range(tIn):
                    visual_list.append(('input_image_%02d'%p, util.tensor2im(input_image[0, p:p+1, :, :, :])))
                    visual_list.append(('input_mask_%02d'%p, util.tensor2mask(input_back_mask[0, p:p+1, :, :, :])))
                    visual_list.append(('input_semantic_%02d'%p, util.tensor2label(input_semantic[0,p:p+1,:,:,:], opt.semantic_nc)))
                    visual_list.append(('input_instance_%02d'%p, util.tensor2mask(input_edge[0,p:p+1,:,:,:])))
                for k in range(tOut):
                    visual_list.append(('label_semantic_%02d'%k, util.tensor2label(label_semantic[0,k:k+1,:,:,:],opt.semantic_nc)))
                    visual_list.append(('warp_image_bwd_%02d'%k, util.tensor2im(warp_image_bwd[0][0, k*3:(k+1)*3, :, :])))
                    visual_list.append(('warp_mask_bwd_%02d'%k, util.tensor2mask(warp_mask_bwd[0][0, k*1:(k+1)*1, :, :])))
                    visual_list.append(('gt_warp_image_%02d'%k, util.tensor2im(gt_warp[0][0,k*3:(k+1)*3, :, :])))
                    visual_list.append(('occlusion_fwd_%02d'%k, util.tensor2occ(occlusion_fwd[0][0, k*1:(k+1)*1, :, :])))
                    visual_list.append(('occlusion_bwd_%02d'%k, util.tensor2occ(occlusion_bwd[0][0, k*1:(k+1)*1, :, :])))
                    visual_list.append(('label_image_%02d'%k, util.tensor2im(label_image[0, k:k+1, :, :, :])))
                    visual_list.append(('label_mask_%02d'%k, util.tensor2mask(label_back_mask[0, k:k+1, :, :, :])))
                visuals = OrderedDict(visual_list)                          
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == 0:
                visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                modelG.module.save('latest')
                modelD.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
           
        # end of epoch 
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
        if epoch > opt.niter:
            modelG.module.update_learning_rate(epoch)
            modelD.module.update_learning_rate(epoch)

def reshape(tensors):
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]
    if tensors is None:
        return None
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)



if __name__ == "__main__":
   train()
