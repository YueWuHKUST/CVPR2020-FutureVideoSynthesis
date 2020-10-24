import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
# Sorted
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def grid_sample(self, input1, input2, method):
        return torch.nn.functional.grid_sample(input1, input2, mode=method, padding_mode='border')

    def resample(self, image, flow, method='bilinear'):        
        b, c, h, w = image.size()        
        if not hasattr(self, 'grid') or self.grid.size() != flow.size():
            self.grid = self.get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
        final_grid = (self.grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
        output = self.grid_sample(image, final_grid, method)
        return output 

    def get_grid(self, batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
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

def make_power_2(n, base=32.0):    
    return int(round(n / base) * base)

def get_img_params(opt, size):
    w, h = size
    if opt.dataset == 'cityscapes':
        new_h = int(opt.loadSize // 2)
        new_w = opt.loadSize
    elif opt.dataset == 'kitti':
        new_h = 256
        new_w = 832
    flip = (random.random() > 0.5)
    if opt.isTrain is True:
        return {'flip': flip, 'new_size': (new_w, new_h)}
    else:
        return {'flip': False, 'new_size': (new_w, new_h)}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    # Data augmentation for dataset except cityscapes
    if opt.isTrain:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))
    else:
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))
    if toTensor:
        transform_list += [transforms.ToTensor()]# divide by 255
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size    
    return img.resize((w, h), method)

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
