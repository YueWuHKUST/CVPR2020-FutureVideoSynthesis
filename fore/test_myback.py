import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0
import torch.nn.functional as F
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer


# Test program for foreground object prediction
# Each sequence return a list
opt = TestOptions().parse()
root  = "/disk2/yue/server6_backup/final/FutureVideoSynthesis/result/%s/"%opt.dataset
if opt.dataset == 'cityscapes':
    height = 512
    width = 1024
else:
    height = 256
    width = 832




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


def mask_clip(mask):
    h, w = mask.shape
    mask_cliped = np.zeros((h, w))
    mask_cliped[mask > 0] = 1
    return mask_cliped

def order_one_sample(whole_image, segs, mask):
    mask = np.tile(np.expand_dims(mask, 0), [3,1,1])
    whole_image = whole_image*(1.0 - mask) + segs * mask
    return whole_image



# All segs data
# each is a 5 frame list
# depth map 1 x 4 x 1 x h x w
# target_back 1 x 5 x 3 x h x w


def combine_segs(opt, all_segs, all_masks, depth_map, target_back_map, input_mask):
    pred_segs_sequence = []
    for id in range(opt.tOut):
        pred_segs_sequence.append(np.zeros((3, height, width)))
    pred_mask_sequence = []
    for id in range(opt.tOut):
        pred_mask_sequence.append(np.zeros((height, width)))
    # Construct segs
    last_depth_map = depth_map[0,-1,0,...]
    for k in range(opt.tOut):
        segs = []
        back_k = target_back_map[0,k,...]
        for p in range(len(all_segs)):
            cnt_seg = all_segs[p][k]
            cnt_mask = all_masks[p][k]
            cnt_input_mask = input_mask[0, p*opt.tIn:(p+1)*opt.tIn, ...]
            cnt_last_mask = cnt_input_mask[-1, ...]
            cnt_depth = np.sum(last_depth_map *cnt_last_mask)/np.sum(cnt_last_mask)
            segs.append((cnt_seg, cnt_mask, cnt_depth))
        segs.sort(key=lambda x: x[2], reverse=True)  # lambda x:x[1]返回list的第二个数据
        #print("Len segs  = ", len(segs))
        for i in range(len(segs)):
            #print("Overlap %d"%i)
            curr_segs = segs[i]
            curr_depth = curr_segs[2]
            #print(cnt_depth)
            curr_pp = curr_segs[0][0,...]
            #print(curr_pp.shape)
            curr_mask = curr_segs[1][0,0,...]
            #print(curr_mask.shape)
            pred_segs_sequence[k] = \
                order_one_sample(pred_segs_sequence[k], curr_pp, curr_mask)
            pred_mask_sequence[k] = pred_mask_sequence[k] + curr_mask
            pred_mask_sequence[k] = mask_clip(pred_mask_sequence[k])
            pred_mask_sequence[k] = pred_mask_sequence[k].astype(np.uint8)
    return pred_segs_sequence, pred_mask_sequence

def combine_fore_back(opt, pred_segs, pred_masks, pred_backs):
    pred_combine = []
    for i in range(opt.tOut):
        cnt_back = pred_backs[0, i, ...]
        cnt_mask = np.tile(np.expand_dims(pred_masks[i], axis=0), [3, 1, 1])
        #print("cnt_mask shape =", cnt_mask.shape)
        #print("cnt_ses shape = ", pred_segs[i].shape)
        #print("cnt_back shape", cnt_back.shape)
        cnt_combine = (1.0 - cnt_mask) * cnt_back + cnt_mask * pred_segs[i]
        pred_combine.append(cnt_combine)
    return pred_combine






def test():
    
    
    ### initialize dataset
    modelG, flowNet = create_model(opt)
    data_loader = CreateDataLoader(opt, flowNet)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#testing %s videos = %d' %(opt.dataset,  dataset_size))

    ### set parameters    
    n_gpus = len(opt.gpu_ids)            # number of gpus used for generator for each batch
    tIn, tOut = opt.tIn, opt.tOut
    channel_all = opt.semantic_nc + opt.image_nc + opt.flow_nc
    semantic_nc = 1
    flow_nc = opt.flow_nc
    image_nc = 3
    back_image_nc = 3
    ### real training starts here  
    for idx, data in enumerate(dataset):
        #if idx < 470:
        #    continue
        input_image = Variable(data['Image'][:, :tIn*3, ...]).view(-1, tIn, 3, height, width)
        input_semantic = Variable(data['Semantic'][:, :tIn*semantic_nc, ...]).view(-1, tIn, semantic_nc, height, width)
        target_back_map = Variable(data['Back'][:, tIn*image_nc:(tIn+tOut)*image_nc, ...]).view(-1, tOut, image_nc, height, width)
        depth_map = Variable(data['Depths'])
        #print("depth_map shape", depth_map.size())
        if opt.dataset == 'cityscapes' and opt.loadSize == 1024:
            depth_map = F.interpolate(depth_map, scale_factor=2.0, mode='bilinear')
        else:
            depth_map = depth_map.view(-1, tIn, 375, 1242)
            depth_map = F.interpolate(depth_map, (width, height), mode='bilinear')
        depth_map = depth_map.view(-1, tIn, 1, height, width).numpy()
        videoid = Variable(data['VideoId'])
        #print(videoid.shape)
        input_semantic = input_semantic.float().cuda()
        target_back_map = target_back_map.float().cuda()
        combine_back = target_back_map


        #print(data['Combine'].size())
        #if len(data['Combine'].size()) == 1:
        #    continue
        num_object = data['Combine'].size()[1]//12
        Classes = Variable(data['Classes']).view(-1, num_object)
        all_segs = []
        all_masks = []

        save_dir_idx = root + "/%04d/"%(videoid)
        if not os.path.exists(save_dir_idx):
            os.makedirs(save_dir_idx)
        print("idx = %d num_object = "%(idx), num_object)
        input_flow, input_conf = compute_flow(input_image, tIn, flowNet)
        for p in range(num_object):
            cnt_class = Classes[:,p].numpy()
            #if p == 0:
            #    global_semantic = input_semantic
            #    global_target_map = target_back_map
            #else:
            #    input_semantic = global_semantic
            #    target_back_map = global_target_map

            input_combine = Variable(data['Combine'][:, p*(tIn*image_nc):(p+1)*tIn*image_nc, ...]).view(-1, tIn, image_nc, height, width)
            input_mask = Variable(data['Mask'][:, p*tIn:(p+1)*tIn, ...]).view(-1, tIn, 1, height, width)
            last_object = Variable(data['LastObject'][:, p*3:(p+1)*3, ...]).view(-1, 3, height, width)
            last_mask = Variable(data['LastMasks'][:, p*1:(p+1)*1, ...]).view(-1, 1, height, width)
            last_object = last_object.float().cuda()
            last_mask = last_mask.float().cuda()

            input_combine = input_combine.float().cuda()
            input_mask = input_mask.float().cuda()
            #print("range combine = ", torch.max(input_combine), torch.min(input_combine))
            #print("range input mask = ", torch.max(input_mask), torch.min(input_mask))
            #print("range last object = ", torch.max(last_object), torch.min(last_object))
            warped_object, warped_mask, affine_matrix, pred_complete = modelG.inference(input_combine, input_semantic, input_flow, input_conf, target_back_map, input_mask, last_object, last_mask)
            #### Save temporal result here
            save_dir_cnt = save_dir_idx + "/%02d/"%p
            print("save_dir_cnt = ", save_dir_cnt)
            if not os.path.exists(save_dir_cnt):
                os.makedirs(save_dir_cnt)
            util.save_cnt_result_and_class(save_dir_cnt, warped_object, warped_mask, affine_matrix, pred_complete, input_combine, input_mask, last_object, cnt_class)

            # changed warped_object, warped_mask to numpy
            warped_object_npy = []
            warped_mask_npy = []
            for k in range(len(warped_object)):
                cnt_object = warped_object[k].cpu().numpy()
                cnt_mask = warped_mask[k].cpu().numpy()
                warped_object_npy.append(cnt_object)
                warped_mask_npy.append(cnt_mask)
            all_segs.append(warped_object_npy)
            all_masks.append(warped_mask_npy)
        # All segs data
        # each is a 5 frame list
        # depth map 1 x 5 x 1 x h x w
        # target_back 1 x 5 x 3 x h x w
        pred_segs, pred_masks = combine_segs(opt, all_segs, all_masks, depth_map, combine_back.cpu().numpy(), Variable(data['Mask']).cpu().numpy())
        pred_combines = combine_fore_back(opt, pred_segs, pred_masks, combine_back.cpu().numpy())
        util.save_pred_all(opt.save_phase, save_dir_idx, pred_segs, pred_masks, pred_combines, opt.which_epoch, False)
        #### Save combine all result here



if __name__ == "__main__":
   test()
