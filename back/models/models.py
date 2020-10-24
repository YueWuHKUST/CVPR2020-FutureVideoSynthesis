import torch.nn as nn
def create_model(opt):
    from .backpred_model_G import BackPredModelG
    modelG = BackPredModelG()
    if opt.isTrain:
        from .backpred_model_D import BackPredModelD
        modelD = BackPredModelD()

    from .pwcnet import PWCNet
    flowNet = PWCNet()
    modelG.initialize(opt)
    if opt.isTrain and len(opt.gpu_ids):
        modelD.initialize(opt)
        flowNet.initialize(opt)        
        modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
        modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
        flowNet = nn.DataParallel(flowNet, device_ids=opt.gpu_ids)
        return [modelG, modelD, flowNet]
    else:
        flowNet.initialize(opt)
        return [modelG, flowNet]
