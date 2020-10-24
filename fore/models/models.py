import torch.nn as nn
def create_model(opt):    
    from .forepred_model_G import ForePredModelG
    modelG = ForePredModelG()
    if opt.isTrain:
        from .forepred_model_D import ForePredModelD
        modelD = ForePredModelD()
    
    from .pwcnet import PWCNet
    flowNet = PWCNet()
        
    modelG.initialize(opt)
    if opt.isTrain and len(opt.gpu_ids):
        modelD.initialize(opt)
        flowNet.initialize(opt)        
        if opt.n_gpus_gen == len(opt.gpu_ids):
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
            modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
            flowNet = nn.DataParallel(flowNet, device_ids=opt.gpu_ids)
        else:             
            if opt.batchSize == 1:
                gpu_split_id = opt.n_gpus_gen + 1
                modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])                
            else:
                gpu_split_id = opt.n_gpus_gen
                modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
            modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids[gpu_split_id:] + [opt.gpu_ids[0]])
            flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        return [modelG, modelD, flowNet]
    else:
        flowNet.initialize(opt)
        return [modelG, flowNet]