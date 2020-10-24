import numpy as np
import torch
import sys
from .base_model import BaseModel#, resample

class PWCNet(BaseModel):
    def name(self):
        return 'PWCNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        from .pwcnet_pytorch.models.PWCNet import pwc_dc_net
        pwc_model_fn = './models/pwcnet_pytorch/pwc_net.pth.tar';
        self.pwcnet = pwc_dc_net(pwc_model_fn)
        self.pwcnet = self.pwcnet.cuda()
        self.pwcnet.eval()
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input_A, input_B):        
        with torch.no_grad():
            size = input_A.size()
            assert(len(size) == 4 or len(size) == 5)
            if len(size) == 5:
                b, n, c, h, w = size
                input_A = input_A.contiguous().view(-1, c, h, w)
                input_B = input_B.contiguous().view(-1, c, h, w)
                flow, conf = self.compute_flow_and_conf(input_A, input_B)
                return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
            else:
                return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        assert(im1.size()[1] == 3)
        assert(im1.size() == im2.size())        
        old_h, old_w = im1.size()[2], im1.size()[3]
        new_h, new_w = old_h//64*64, old_w//64*64
        if old_h != new_h:
            downsample = torch.nn.Upsample(size=(new_h, new_w), mode='bilinear')
            im1 = downsample(im1)
            im2 = downsample(im2)
        upsample = torch.nn.Upsample(size=(old_h, old_w), mode='bilinear')
        #self.flowNet.cuda(im1.get_device())
        #data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)
        data1 = torch.cat([im1, im2], dim=1)
        data1 = (data1 + 1.0)/2.0
        #print("data1 size =", data1.size())
        flow1 = self.pwcnet(data1)
        flow1 = flow1*20.0
        #print("im1", im1.size(),im2.size(),flow1.size())
        #print("warp", self.resample(im2, flow1).size())
        #if old_h != new_h:
        flow1 = upsample(flow1) * old_h / new_h
            #conf = upsample(conf)
        #print("im1", im1.size(),im2.size(),flow1.size())
        conf = (self.norm(im1 - self.resample(im2, flow1)) < 0.02).float()
        
        return flow1.detach(), conf.detach()

    def norm(self, t):
        return torch.sum(t*t, dim=1, keepdim=True)   
