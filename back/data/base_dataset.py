import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def make_power_2(n, base=32.0):    
    return int(round(n / base) * base)

def get_img_params(opt, size):
    w, h = size
    if opt.dataset == 'cityscapes':
        # For cityscapes, only do flipping
        new_h = int(opt.loadSize // 2)
        new_w = opt.loadSize
        flip = (random.random() > 0.5)
        if opt.isTrain is True:
            return {'flip': flip}
        else:
            return {'flip': False}
    elif opt.dataset == 'kitti':
        new_h = 256
        new_w = 832
    # For dataset except cityscapes, do resize or random crop augmentation
    if opt.dataset != 'cityscapes':
        crop_x = 0
        crop_y = 0
        if opt.isTrain is True:
            resize_or_crop = (random.random() > 0.5)
            if resize_or_crop is False:
                crop_x = random.randint(0, np.maximum(0, w - new_w))
                crop_y = random.randint(0, np.maximum(0, h - new_h))     
            flip = (random.random() > 0.5)
        else:
            resize_or_crop = True
            flip = False
        return {'resize_or_crop': resize_or_crop, 'new_size': (new_w, new_h), 'crop_pos': (crop_x, crop_y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    # Data augmentation for dataset except cityscapes
    if opt.isTrain:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
        if opt.dataset != 'cityscapes':
            if params['resize_or_crop'] is True:
                ### resize input image
                transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))
            else:
                ### crop patches from image
                transform_list.append(transforms.Lambda(lambda img: __crop(img, params['new_size'], params['crop_pos'])))
    else:
        # Test, resize images
        if opt.dataset != 'cityscapes':
            transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))
    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size    
    return img.resize((w, h), method)

def __crop(img, size, pos):
    ow, oh = img.size
    tw, th = size
    x1, y1 = pos        
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, min(ow, x1 + tw), min(oh, y1 + th)))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
