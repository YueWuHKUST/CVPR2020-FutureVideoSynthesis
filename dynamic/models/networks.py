### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):        
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, tOut, ngf, which_model_netG, n_downsampling, norm, scale, is_Train, gpu_ids=[], opt=[]):
    # Default generator mode, composite
    norm_layer = get_norm_layer(norm_type=norm)

    netG = ResNetGenerator(input_nc, output_nc, tOut, is_Train, ngf, n_downsampling, opt.n_blocks_local, norm_layer)
    #print_network(netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# Get grid for resample

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

##############################################################################
# Classes
##############################################################################

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





class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

def resize_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

def downconv(in_chnls, out_chnls, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_chnls, out_chnls, kernel_size,
                  stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chnls, out_chnls, kernel_size,
                  stride=1, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )

def conv(in_chnls, out_chnls):
    return nn.Sequential(
        nn.Conv2d(in_chnls, out_chnls, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True)
    )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,10,kernel_size=3,stride=1,padding=1,bias=True)


def conv_pwcnet(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

### Backbone use resent50
### Following the idea of pwcnet/flownet2
### Input encoder feature, last predict flow, \
### Can't add correlation between warped frame and gt, can't input gt
#, features of last input
class ResNetGenerator(BaseNetwork):
    def __init__(self, input_nc, output_nc, tOut, is_Train, ngf, n_downsampling, n_blocks,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        # ngf number of generator fileters in first conv layer
        '''
        :param input_nc: Input channels for each frames times frame number
        :param output_nc: Output channels for predict next frame
        :param ngf:number of generator filters in first conv layer
        :param n_downsampling:
        :param n_blocks:
        :param norm_layer:
        :param padding_type:
        '''
                # Network sturcture
        # Downsample
        # Conv input->128 7
        # Conv 128->256 3/2/1
        # Conv 256->512 3/2/1
        # Conv 512->1024 3/2/1
        # 9 ResNet Block
        # Upconv
        # Deconv1 1024->512
        # Deconv2 512->256
        # Deconv3 256->128
        ### Downsample input data to get features
        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()        
        #self.resample = Resample2d()
        self.n_downsampling = n_downsampling
        activation = nn.LeakyReLU(True)
        self.tOut = tOut
        self.is_Train = is_Train
        ### Downsample input data to get features

        #self.downconv_high = nn.Sequential(nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=2, padding=1), \
        self.downconv_high = nn.Sequential(nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1), \
                        norm_layer(ngf), activation)
                        #norm_layer(input_nc), activation)

        model_down_input = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        
        

        self.downconv1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * 2), activation)
        self.downconv2 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * 4), activation)
        self.downconv3 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * 8), activation)

        #self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.LeakyRELU = nn.LeakyReLU(0.1)

        mult = 2**n_downsampling
        model_res_down = []
        for i in range(n_blocks - n_blocks//2):
            model_res_down += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model_down_feat = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_feat += copy.deepcopy(model_down_input[4:])

        #### Output all at once
        model_res_up = []
        for i in range(n_blocks//2):
            model_res_up += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        
        self.upconv1 = nn.Sequential(nn.Conv2d(ngf * 12, ngf * 4, kernel_size=3, stride=1, padding=1), norm_layer(ngf * 4), activation)
        self.upconv2 = nn.Sequential(nn.Conv2d(ngf * 6 + 1, ngf * 2, kernel_size=3, stride=1, padding=1), norm_layer(ngf * 2), activation)
        self.upconv3 = nn.Sequential(nn.Conv2d(ngf * 3 + 1, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation)
        
        self.upconv_high = nn.Sequential(nn.Conv2d(ngf * 2 + 1, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation)

        
        model_final_flow_high = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model_final_flow_high = nn.Sequential(*model_final_flow_high)

        model_final_flow_1 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model_final_flow_1 = nn.Sequential(*model_final_flow_1)

        model_final_flow_2 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, 1, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model_final_flow_2 = nn.Sequential(*model_final_flow_2)

        model_final_flow_3 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*4, 1, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model_final_flow_3 = nn.Sequential(*model_final_flow_3)

        self.model_down_input = nn.Sequential(*model_down_input)
        self.model_res_down = nn.Sequential(*model_res_down)
        self.model_res_up = nn.Sequential(*model_res_up)

    def forward(self, loadsize, image, semantic, flow, conf, edge):
        # 1 x 69 x 256 x 512
        input = torch.cat([image, semantic, flow, conf, edge], dim=1)
        #print("input", input.size())
        # 1 x 128 x 256 x 512 
        if loadsize == 1024:
            down_high = self.model_down_input(input)
            #print("down high = ", down_high.size())
            down1 = self.downconv_high(down_high)
            #print("down 1",down1.size())
            #down1 = self.model_down_input(down_high)
        else:
            down1 = self.model_down_input(input)
        # 1 x 256 x 128 x 256
        down2 = self.downconv1(down1)
        #print("down2", down2.size())
        # 1 x 512 x 64 x 128
        down3 = self.downconv2(down2)
        #print("down3", down3.size())
        # 1 x 1024 x 32 x 64
        down4 = self.downconv3(down3)
        #print("down4", down4.size())
        # 1 x 1024 x 32 x 64
        feat = self.model_res_up(self.model_res_down(down4))
        up4 = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        up4_concat = torch.cat([up4, down3], dim=1)
        #print("up4_concat", up4_concat.size())
        # 1 x 512 x 64 x 128
        up4_conv = self.upconv1(up4_concat)

        # 1 x 20 x 64 x 128
        #print("up4_conv", up4_conv.size())
        flow3 = self.model_final_flow_3(up4_conv)
        # 1 x 20 x 128 x 256
        flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=False)
        #print("up4_conv", up4_conv.size())
        # 1 x 512 x 128 x 256
        up3 = F.interpolate(up4_conv, scale_factor=2, mode='bilinear', align_corners=False)
        #print("up3", up3.size())
        # 1 x 768 x 128 x 256
        up3_concat = torch.cat([up3, down2, flow3_up], dim=1)
        # 1 x 256 x 128 x 256
        up3_conv = self.upconv2(up3_concat)
        # 1 x 20 x 128 x 256
        flow2 = self.model_final_flow_2(up3_conv)
        # 1 x 20 x 256 x 512
        flow2_up = F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=False)
        # 1 x 256 x 256 x 512
        up2 = F.interpolate(up3_conv, scale_factor=2, mode='bilinear', align_corners=False)
        # 1 x 384 x 256 x 512
        up2_concat = torch.cat([up2, down1, flow2_up], dim=1)
        up2_conv = self.upconv3(up2_concat)
        # 1 x 20 x 256 x 512
        flow1 = self.model_final_flow_1(up2_conv)
        return flow1
                
                
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out