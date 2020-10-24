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

def compute_flow_pair(a, b, flowNet):
	flow, conf = flowNet(a, b)
	#flow = flow.unsqueeze(1)
	#conf = conf.unsqueeze(1)
	return flow, conf

def inst_and_dynamic(inst_map, dynamic, thre):
	dynamic = (dynamic > 0.5).astype(np.int32)
	dynamic_mask = np.zeros_like(dynamic)
	inst_idx = np.unique(inst_map)
	#maximum is 255 indicate invalid
	for i in range(len(inst_idx)-1):
		cnt_mask = (inst_map == inst_idx[i]).astype(np.int32)
		whether_dynamic = np.sum(dynamic*cnt_mask)/float(np.sum(cnt_mask))
		if whether_dynamic > thre:
			dynamic_mask[cnt_mask == 1] = 1
	return dynamic_mask

def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
	hor = torch.linspace(-1.0, 1.0, cols)
	hor.requires_grad = False
	hor = hor.view(1, 1, 1, cols)
	hor = hor.expand(batchsize, 1, rows, cols)
	ver = torch.linspace(-1.0, 1.0, rows)
	ver.requires_grad = False
	ver = ver.view(1, 1, rows, 1)
	ver = ver.expand(batchsize, 1, rows, cols)

	t_grid = torch.cat([hor, ver], 1)
	t_grid.requires_grad = False

	if dtype == torch.float16: t_grid = t_grid.half()
	return t_grid.cuda(gpu_id)

def grid_sample(input1, input2, mode='bilinear'):    
	return torch.nn.functional.grid_sample(input1, input2, mode=mode, padding_mode='border')

def resample(image, flow, mode='bilinear'):        
	b, c, h, w = image.size()        
	grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
	flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
	#print(flow.size())
	final_grid = (grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
	#print("final_grid", final_grid.size())
	output = grid_sample(image, final_grid, mode)
	return output

def gen_pre_mask(image, flownet, mask):
	mask = torch.from_numpy(mask).float().cuda()
	last = image[:, -1, ...]

	pre_list = []
	for i in range(3):
		pre = image[:, i, ...]
		flo, con = compute_flow_pair(pre, last, flownet)
		prev_mask = resample(mask, flo, mode='nearest')
		pre_list.append(prev_mask)
	return pre_list


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
	
	modelG, flowNet = create_model(opt)

	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	visualizer = Visualizer(opt)
	iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
	save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	print('Doing %d frames' % len(dataset))

	for i, data in enumerate(dataset):
		#if i <= 480:#opt.how_many:
		#	continue
		if opt.dataset == 'cityscapes':
			for idx in range(0, 1):
				print("Testing %04d %02d"%(i, idx))
				_, _, height, width = data['Image'].size()
				t_all = tIn
				t_len = t_all
				input_image = Variable(data['Image'][:, idx*image_nc:(idx+tIn)*image_nc, ...]).view(-1, tIn, image_nc, height, width)
				input_semantic = Variable(data['Semantic'][:, idx*semantic_nc:(idx+tIn)*semantic_nc, ...]).view(-1, tIn, semantic_nc, height, width)
				input_instance = Variable(data['Instance'][:, idx*instance_nc:(idx+tIn)*instance_nc, ...]).view(-1, tIn, 1, height, width)
				input_semantic = input_semantic.float().cuda()
				input_image = input_image.float().cuda()
				input_instance = input_instance.float().cuda()

				input_flow, input_conf = compute_flow(input_image, tIn, flowNet)
					
				pred_dynamic = \
					modelG.inference(input_image, input_semantic, input_flow, input_conf, input_instance)
				# Batch x 1 x h x w
				# Batch x 1 x h x w 
				combine_dynamic = inst_and_dynamic(input_instance[:,-1,...].cpu().float().numpy(), pred_dynamic.cpu().float().numpy(), 0.1)
				
				## Generate previous masks
				previous_mask = gen_pre_mask(input_image, flowNet, combine_dynamic)
				masks = []
				for k in range(3):
					cnt_mask = inst_and_dynamic(input_instance[:, k, ...].cpu().float().numpy(), previous_mask[k].cpu().float().numpy(), 0.1)
					masks.append(cnt_mask)
				visual_list = []
				#visual_list.append(('pred_mask', util.tensor2mask(pred_dynamic[0,...], normalize=False)))
				for p in range(3):
					visual_list.append(('pred_dynamic_%02d'%(idx + p), util.numpy2mask(masks[p][0,...], normalize=False)))
				#visual_list.append(('pred_mask_%02d'%(idx + 3), util.tensor2mask(pred_dynamic[0,...], normalize=False)))
				visual_list.append(('pred_dynamic_%02d'%(idx + 3), util.numpy2mask(combine_dynamic[0,...], normalize=False)))
				visuals = OrderedDict(visual_list)                          
				visualizer.save_test_images(save_dir, visuals, i)
		else:
			print("Testing %04d"%i)
			_, _, height, width = data['Image'].size()
			t_all = tIn
			t_len = t_all
			input_image = Variable(data['Image']).view(-1, tIn, image_nc, height, width)
			input_semantic = Variable(data['Semantic']).view(-1, tIn, semantic_nc, height, width)
			input_instance = Variable(data['Instance']).view(-1, tIn, 1, height, width)
			input_semantic = input_semantic.float().cuda()
			input_image = input_image.float().cuda()
			input_instance = input_instance.float().cuda()

			input_flow, input_conf = compute_flow(input_image, tIn, flowNet)
				
			pred_dynamic = \
				modelG.inference(input_image, input_semantic, input_flow, input_conf, input_instance)
			# Batch x 1 x h x w
			# Batch x 1 x h x w 
			combine_dynamic = inst_and_dynamic(input_instance[:,-1,...].cpu().float().numpy(), pred_dynamic.cpu().float().numpy(), 0.5)
			## Generate previous masks
			previous_mask = gen_pre_mask(input_image, flowNet, combine_dynamic)
			masks = []
			for k in range(3):
				cnt_mask = inst_and_dynamic(input_instance[:, k, ...].cpu().float().numpy(), previous_mask[k].cpu().float().numpy(), 0.5)
				masks.append(cnt_mask)


			visual_list = []
			for p in range(3):
				visual_list.append(('pred_dynamic_%02d'%p, util.numpy2mask(masks[p][0,...], normalize=False)))

			visual_list.append(('pred_dynamic_03', util.numpy2mask(combine_dynamic[0,...], normalize=False)))
			visuals = OrderedDict(visual_list)                          
			visualizer.save_test_images(save_dir, visuals, i)



if __name__ == "__main__":
   test()
