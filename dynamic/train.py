import sys 
#sys.settrace()
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

# Network to detect moving or static object
# Input 4 frames and flow and inst edges and semantic maps
# to detect whether objects are moving
# During training, we use dataset http://deepmotion.cs.uni-freiburg.de/
# And we inference the model to the whole data to generate dynamic masks

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

def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1    
        opt.nThreads = 1
    if opt.dataset == 'cityscapes':
        height = int(opt.loadSize/2)
        width = opt.loadSize
        if opt.loadSize == 1024:
            opt.ImagesRoot = '/disk1/yue/cityscapes/cityscapes/leftImg8bit_sequence_512p/'
    if opt.dataset == 'kitti':
        height = 256
        width = 512
        opt.ImagesRoot = "/disk/clibt/kitti/raw_data/2011_09_26/"
        opt.SemanticRoot = "/disk/clibt/kitti/semantic/pred/2011_09_26/"
        opt.InstanceRoot = "/disk/clibt/kitti/instance/"


    ### initialize models
    modelG, flowNet = create_model(opt)
    visualizer = Visualizer(opt)


    ### Define loss function
    criterionmask = torch.nn.BCELoss()

    ### initialize dataset
    data_loader = CreateDataLoader(opt, flowNet)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training %s videos = %d' % (opt.dataset, dataset_size))
    

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
    else:    
        start_epoch, epoch_iter = 1, 0

    ### set parameters    
    n_gpus = 1 #opt.n_gpus_gen // opt.batchSize             # number of gpus used for generator for each batch
    tIn = opt.tIn
    channel_all = opt.semantic_nc + opt.image_nc + opt.flow_nc
    semantic_nc = 1
    flow_nc = opt.flow_nc
    image_nc = opt.image_nc
    instance_nc = 1
    
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    total_steps = total_steps // opt.print_freq * opt.print_freq  

    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for idx, data in enumerate(dataset, start=epoch_iter):        
            #i+=1
            if total_steps % opt.print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0
        
            input_image = Variable(data['Image']).view(-1, tIn, image_nc, height, width)
            input_semantic = Variable(data['Semantic']).view(-1, tIn, semantic_nc, height, width)
            input_instance = Variable(data['Instance']).view(-1, tIn, 1, height, width)
            label_dynamic = Variable(data['Dynamic']).view(-1, 1, 1, height, width)
            print("Loaded data success")
            input_semantic = input_semantic.float().cuda()
            input_image = input_image.float().cuda()
            input_instance = input_instance.float().cuda()
            label_dynamic = label_dynamic.float().cuda()
       
            input_flow, input_conf = compute_flow(input_image, tIn, flowNet)
            print("Compute flow success")
            #print("input_image",input_image.size(),input_semantic.size(),input_back_mask.size(),input_edge.size(),input_flow.size(),input_conf.size())
            pred_dynamic = \
                modelG(input_image, input_semantic, input_flow, input_conf, input_instance)
                
            _, _, _, height, width = label_mask.size()
            label_mask = label_dynamic.view(-1, height * width)
            pred_mask = pred_dynamic.view(-1, height * width)
            loss_G = criterionmask(pred_mask, label_mask)    

            ###################################### Backward Pass ################################# 
            optimizer_G = modelG.module.optimizer_G

            # update generator weights
            optimizer_G.zero_grad()            
            loss_G.backward()        
            optimizer_G.step()

            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                #print("pred_semantic", pred_semantic[0].size())
                visual_list = []
                visual_list.append(('label_dynamic', util.tensor2mask(label_dynamic[0,...], normalize=False)))
                visual_list.append(('pred_mask', util.tensor2mask(pred_dynamic[0,...], normalize=False)))
                visuals = OrderedDict(visual_list)                          
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == 0:
                visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                modelG.module.save('latest')
                
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
           
        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            modelG.module.save('latest')
            
            modelG.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            modelG.module.update_learning_rate(epoch)


def reshape(tensors):
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]
    if tensors is None:
        return None
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)



if __name__ == "__main__":
   train()
