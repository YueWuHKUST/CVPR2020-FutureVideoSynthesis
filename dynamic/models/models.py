### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch.nn as nn
def create_model(opt):              
    from .dynamic_detect import DynamicDetect
    modelG = DynamicDetect()


    from .pwcnet import PWCNet
    flowNet = PWCNet()
    
    modelG.initialize(opt)
    if opt.isTrain and len(opt.gpu_ids):
        flowNet.initialize(opt)        
        modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
        flowNet = nn.DataParallel(flowNet, device_ids=opt.gpu_ids)
        return [modelG, flowNet]
    else:
        flowNet.initialize(opt)
        return [modelG, flowNet]
