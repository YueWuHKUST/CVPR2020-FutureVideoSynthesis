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

# Train script for dynamic motion prediction model



# Predict future 5/10 frames at 256x512 / 512 x 1024 resolution


def compute_flow(input_images, tIn, flowNet):
    input_images = input_images.data.cuda()
    input_flow = [None]*(tIn-1)
    input_conf = [None]*(tIn-1)
    for i in range(tIn - 1):
        input_image_a = input_images[:,i,:,:,:]
        input_image_b = input_images[:,i+1,:,:,:]
        out_flow, out_conf = flowNet(input_image_a, input_image_b)
        out_flow = out_flow.unsqueeze(1)
        out_conf = out_conf.unsqueeze(1)
        input_flow[i], input_conf[i] = out_flow, out_conf
    input_flow_cat = torch.cat([input_flow[k] for k in range(tIn - 1)], dim=1)
    input_conf_cat = torch.cat([input_conf[k] for k in range(tIn - 1)], dim=1)
    return input_flow_cat, input_conf_cat

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
        opt.ImagesRoot = '/disk1/yue/dataset/leftImg8bit_sequence_512p/'
        opt.BackRoot = '/disk1/yue/dataset/leftImg8bit_sequence_512p_background_inpainted/'
        opt.SemanticRoot = '/disk1/yue/dataset/semantic_new/'
        opt.Instance = '/disk1/yue/dataset/instance_upsnet/train/'
        opt.niter = 100
        opt.niter_decay = 100

    ### initialize dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training Cityscapes videos = %d' % dataset_size)

    ### initialize models
    modelG, modelD, flowNet = create_model(opt)
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
    total_steps = 0
    all_loss = np.zeros((110000, 20))
    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()

        for idx, data in enumerate(dataset, start=epoch_iter):
            total_steps += opt.batchSize
            #print("idx = ", idx)
            save_fake = total_steps % opt.display_freq == 0
            if total_steps % opt.print_freq == 0:
                iter_start_time = time.time()

            epoch_iter += opt.batchSize
            _, n_frames_total, height, width = data['Image'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1
            
            ### Input here
            input_image = Variable(data['Image'][:, :tIn*3, ...]).view(-1, tIn, 3, height, width)
            input_semantic = Variable(data['Semantic'][:, :tIn*semantic_nc, ...]).view(-1, tIn, semantic_nc, height, width)
            input_combine = Variable(data['Combine'][:, :tIn*image_nc, ...]).view(-1, tIn, image_nc, height, width)
            input_flow, input_conf = compute_flow(input_image, tIn, flowNet)
            target_back_map = Variable(data['Back'][:, tIn*image_nc:(tIn+tOut)*image_nc, ...]).view(-1, tOut, image_nc, height, width)
            input_mask = Variable(data['Mask'][:, :tIn*1, ...]).view(-1, tIn, 1, height, width)
            last_object = Variable(data['LastObject']).view(-1, 3, height, width)


            ### Label for loss here
            label_combine = Variable(data['Combine'][:, tIn*image_nc:(tIn+tOut)*image_nc, ...]).view(-1, tOut, image_nc, height, width)
            label_mask = Variable(data['Mask'][:, (tIn)*1:(tIn+tOut)*1, ...]).view(-1, tOut, 1, height, width)

            input_semantic = input_semantic.float().cuda()
            input_combine = input_combine.float().cuda()
            target_back_map = target_back_map.float().cuda()
            input_mask = input_mask.float().cuda()
            last_object = last_object.float().cuda()
            label_combine = label_combine.float().cuda()
            label_mask = label_mask.float().cuda()

            warped_object, warped_mask, affine_matrix, pred_complete = modelG(input_combine, input_semantic, input_flow, input_conf, target_back_map, input_mask, last_object)
            losses = modelD(0, [warped_object, warped_mask, affine_matrix, pred_complete, label_combine, label_mask])

            real_sequence, fake_sequence = modelD.module.gen_seq(input_mask, warped_mask, label_mask, tIn, tOut)
            losses_T = modelD(1, [real_sequence, fake_sequence])

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
                visual_list.append(('last_object', util.tensor2im(last_object[0,...])))
                for k in range(tOut):
                    visual_list.append(('pred_combine_%02d' % k, util.tensor2im(pred_complete[k][0, ...])))
                    visual_list.append(('label_combine_%02d' % k, util.tensor2im(label_combine[0,k, ...])))
                    visual_list.append(('label_mask_%02d' % k, util.tensor2mask(label_mask[0,k, ...])))
                    visual_list.append(('target_back_%02d' % k, util.tensor2im(target_back_map[0,k, ...])))
                    visual_list.append(('pred_mask_%02d'%k, util.tensor2mask(warped_mask[k][0,...])))
                for k in range(tIn):
                    visual_list.append(('input_combine_%02d' % k, util.tensor2im(input_combine[0, k, ...])))
                    visual_list.append(('input_mask_%02d'%k, util.tensor2mask(input_mask[0, k, ...])))
                    visual_list.append(('input_semantic_%02d' % k, \
                                                util.tensor2label(input_semantic[0, k, ...], opt.semantic_nc)))
                visuals = OrderedDict(visual_list)
                visualizer.display_current_results(visuals, epoch, total_steps)


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
                #errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                errors = {k: v.data.item() for k, v in loss_dict.items()}
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, all_loss)
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

        ### finetune all scales
        if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            modelG.module.update_fixed_params()

if __name__ == "__main__":
   train()
