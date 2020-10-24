import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy


#from .base_model import resample
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

def define_G(input_nc, output_nc, tOut, ngf, n_downsampling, norm, scale, is_Train, gpu_ids=[], opt=[]):   
    norm_layer = get_norm_layer(norm_type=norm)
    netG = ResNetGenerator(input_nc, output_nc, tOut, is_Train, ngf, n_downsampling, opt.n_blocks_local, norm_layer)

    #print_network(netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', num_D=1, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, num_D, False)   
    print_network(netD)
    if len(gpu_ids) > 0:    
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    #print('Total number of parameters: %d' % num_params)

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

resample_method = 'border'
#;border'

def grid_sample(input1, input2, mode='bilinear'):    
    return torch.nn.functional.grid_sample(input1, input2, mode=mode, padding_mode=resample_method)

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

    def grid_sample(self, input1, input2, mode):
        return torch.nn.functional.grid_sample(input1, input2, mode=mode, padding_mode=resample_method)

    def resample(self, image, flow, mode='bilinear'):        
        b, c, h, w = image.size()        
        if not hasattr(self, 'grid') or self.grid.size() != flow.size():
            self.grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
        final_grid = (self.grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
        output = self.grid_sample(image, final_grid, mode='bilinear')
        return output


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
        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()        
        self.n_downsampling = n_downsampling
        activation = nn.LeakyReLU(True)
        self.tOut = tOut
        self.is_Train = is_Train
        ### Downsample input data to get features
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

        #### Output all at once
        model_res_up = []
        for i in range(n_blocks//2):
            model_res_up += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        
        self.upconv1 = nn.Sequential(nn.Conv2d(ngf * 12, ngf * 4, kernel_size=3, stride=1, padding=1), norm_layer(ngf * 4), activation)
        self.upconv2 = nn.Sequential(nn.Conv2d(ngf * 6 + 4 * tOut, ngf * 2, kernel_size=3, stride=1, padding=1), norm_layer(ngf * 2), activation)
        self.upconv3 = nn.Sequential(nn.Conv2d(ngf * 3 + 4 * tOut, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation)

        model_final_flow_1 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, tOut * 4, kernel_size=7, padding=0)]
        self.model_final_flow_1 = nn.Sequential(*model_final_flow_1)

        model_final_flow_2 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, tOut * 4, kernel_size=7, padding=0)]
        self.model_final_flow_2 = nn.Sequential(*model_final_flow_2)

        model_final_flow_3 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*4, tOut * 4, kernel_size=7, padding=0)]
        self.model_final_flow_3 = nn.Sequential(*model_final_flow_3)

        model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, tOut*1, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model_down_input = nn.Sequential(*model_down_input)
        self.model_res_down = nn.Sequential(*model_res_down)
        self.model_res_up = nn.Sequential(*model_res_up)

    def forward(self, loadsize, image, semantic, flow, conf, edge, backmask_in):
        # 1 x 69 x 256 x 512
        
        input = torch.cat([image, semantic, flow, conf, edge, backmask_in], dim=1)

        down1 = self.model_down_input(input)

        down2 = self.downconv1(down1)

        down3 = self.downconv2(down2)


        down4 = self.downconv3(down3)

        feat = self.model_res_up(self.model_res_down(down4))

        up4 = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        
        up4_concat = torch.cat([up4, down3], dim=1)
 
        up4_conv = self.upconv1(up4_concat)


        flow3 = self.model_final_flow_3(up4_conv)

        flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=False)

        up3 = F.interpolate(up4_conv, scale_factor=2, mode='bilinear', align_corners=False)

        up3_concat = torch.cat([up3, down2, flow3_up], dim=1)

        up3_conv = self.upconv2(up3_concat)

        flow2 = self.model_final_flow_2(up3_conv)

        flow2_up = F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=False)
        
        up2 = F.interpolate(up3_conv, scale_factor=2, mode='bilinear', align_corners=False)
       
        up2_concat = torch.cat([up2, down1, flow2_up], dim=1)
        up2_conv = self.upconv3(up2_concat)

        flow1 = self.model_final_flow_1(up2_conv)

        return [flow1, flow2, flow3], up2_conv, input#, conf_map


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

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, norm_layer,
                                       getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]            
            for i in range(len(model)):
                result.append(model[i](result[-1]))            
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))                                
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)                    
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)            

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, mask):
        while x.size()[3] > 600:
            x, y = self.downsample(x), self.downsample(y)
        #print("x =", x.size())
        #print("y =", y.size())
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            _, ch, h, w = x_vgg[i].size()
            curr_mask = F.upsample_nearest(mask, (h, w))
            curr_mask = curr_mask.expand(-1, ch, -1, -1)
            loss += self.weights[i] * self.criterion(x_vgg[i]*curr_mask, y_vgg[i].detach()*curr_mask)
        return loss

class MaskTwoL1Loss(nn.Module):
    def __init__(self):
        super(MaskTwoL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask, backmask):
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask * backmask, target * mask * backmask)
        return loss

class MaskOneL1Loss(nn.Module):
    def __init__(self):
        super(MaskOneL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, backmask):
        #mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * backmask, target * backmask)
        return loss

class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        #self.weights = [0.5, 1, 2, 8, 32]
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, input, target, mask=None):
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, input.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:                
                loss += self.weights[i] * self.criterion(input * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights)-1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def gradient_x(self, img, backmask):
        gx = (img[:,:,:-1,:] - img[:,:,1:,:])*backmask[:,:,1:,:]
        return gx

    def gradient_y(self, img, backmask):
        gy = (img[:,:,:,:-1] - img[:,:,:,1:])*backmask[:,:,:,1:]
        return gy

    def compute_smooth_loss(self, flow_x, img, backmask):
        flow_gradients_x = self.gradient_x(flow_x, backmask)
        flow_gradients_y = self.gradient_y(flow_x, backmask)

        image_gradients_x = self.gradient_x(img, backmask)
        image_gradients_y = self.gradient_y(img, backmask)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, True))

        smoothness_x = flow_gradients_x * weights_x
        smoothness_y = flow_gradients_y * weights_y

        return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img, backmask):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(flow[:,i:i+1,:,:], img, backmask)
        return smoothness/2

    def forward(self, flow, image, mask):
        return self.compute_flow_smooth_loss(flow, image, mask)



from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Consistency(nn.Module):
    # Consistency loss for a pair of optical flow
    def __init__(self):
        super(Consistency, self).__init__()
        self.beta = 0.05
        self.weight = 0.02

    def L2_norm(self, x): 
        return F.normalize(x, p=2, dim=1, eps=1e-12)#.unsqueeze(1)
        #max value

    def forward(self, flow_fwd, flow_bwd, s, bwd_valid_region, fwd_valid_region):
        devide = flow_fwd.get_device()
        alpha = torch.FloatTensor([3.0]).cuda(devide)
        #print("flow_fwd ", flow_fwd.size())
        bwd2fwd_flow_pyramid = resample(flow_bwd, flow_fwd)# From bwd coordinate to src coordinate
        fwd2bwd_flow_pyramid = resample(flow_fwd, flow_bwd)# From fwd coordinate to tgt coordinate
        #print("bwd2fwd_flow_pyramid", bwd2fwd_flow_pyramid.size())
        fwd_diff = torch.abs(bwd2fwd_flow_pyramid + flow_fwd)# In src
        bwd_diff = torch.abs(fwd2bwd_flow_pyramid + flow_bwd)# In tgt
        #print("fwd_diff size = ", fwd_diff.size())
        fwd_consist_bound = self.beta * self.L2_norm(flow_fwd) * 2**s
        bwd_consist_bound = self.beta * self.L2_norm(flow_bwd) * 2**s
        #print("fwd_consist_bound = ", fwd_consist_bound.size())
        fwd_consist_bound = torch.max(fwd_consist_bound, alpha).clone().detach()
        bwd_consist_bound = torch.max(bwd_consist_bound, alpha).clone().detach()
        fwd_mask = (fwd_diff * 2**s < fwd_consist_bound).float()# In src
        fwd_mask *= fwd_valid_region
        bwd_mask = (bwd_diff * 2**s < bwd_consist_bound).float()# In tgt
        bwd_mask *= bwd_valid_region

        #print("fwd_diff size = ", fwd_diff.size())
        #print("fwd_mask = ", fwd_mask.size())
        flow_consistency_loss = self.weight/2 * \
            (torch.sum(torch.mean(fwd_diff, dim=1, keepdim=True) * fwd_mask) / torch.sum(fwd_mask) + \
            torch.sum(torch.mean(bwd_diff, dim=1, keepdim=True) * bwd_mask) / torch.sum(bwd_mask))
        #print(flow_consistency_loss.size())
        #print("Consistency loss = ", flow_consistency_loss)

        return fwd_mask, bwd_mask, flow_consistency_loss
