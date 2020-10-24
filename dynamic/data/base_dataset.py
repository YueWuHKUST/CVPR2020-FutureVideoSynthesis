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
        if opt.loadSize == 512:
            new_w = 512
            new_h = 256
        else:
            new_w = 1024
            new_h = 512
    elif opt.dataset == 'kitti': 	
        new_w = 512
        new_h = 256

    if opt.isTrain is True:
        if opt.data_augment:
            resize_or_crop = random.random() > 0.5
            if resize_or_crop is False: 
                x_span = (w - new_w) // 2
                crop_x = np.maximum(0, np.minimum(x_span*2, int(np.random.randn() * x_span/3 + x_span)))        
                crop_y = random.randint(0, np.minimum(np.maximum(0, h - new_h), new_h // 8))
            flip = random.random() > 0.5
        else:
            crop_x = 0
            crop_y = 0
            resize_or_crop = True
            flip = False
    else:
        crop_x = 0
        crop_y = 0
        resize_or_crop = True
        flip = False
    return {'new_size': (new_w, new_h), 'crop_size': (new_w, new_h), 'crop_pos': (crop_x, crop_y), 'flip': flip, 'resize_or_crop': resize_or_crop}



def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    ### resize input image
    if params['resize_or_crop'] is True:
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))

    if params['resize_or_crop'] is False:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_size'], params['crop_pos'])))    

    ### random flip
    if opt.isTrain:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def toTensor_normalize():    
    transform_list = [transforms.ToTensor()]    
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

def concat_frame(A, Ai, nF):
    if A is None:
        A = Ai
    else:
        c = Ai.size()[0]
        if A.size()[0] == nF * c:
            A = A[c:]
        A = torch.cat([A, Ai])
    return A