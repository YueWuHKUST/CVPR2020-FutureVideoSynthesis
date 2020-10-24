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


def define_G(input_nc, dataset, loasSize, tOut, ngf, gpu_ids=[]):
    # Default generator mode, composite
    norm_layer = get_norm_layer(norm_type='instance')
    netG = STN(input_nc, dataset, loasSize, tOut, ngf, norm_layer)

    # print_network(netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type='instance')
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, num_D, getIntermFeat)
    # print_network(netD)
    if len(gpu_ids) > 0:
        netD.cuda(gpu_ids[0])
        # netD.cuda(gpu_ids[1])
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################
class STN(nn.Module):
    def __init__(self, input_nc, dataset, loadSize, tOut, ngf, norm_layer=nn.BatchNorm2d):
        # ngf number of generator fileters in first conv layer
        '''
        :param input_nc: Input channels for each frames times frame number
        :param output_nc: Output channels for predict next frame
        :param ngf:number of generator filters in first conv layer
        :param n_downsampling:3
        :param n_blocks:
        :param norm_layer:
        :param padding_type:
        '''
        
        super(STN, self).__init__()
        activation = nn.LeakyReLU(True)
        self.tOut = tOut
        ### Downsample input data to get features
        encoder = []
        # Conv0
        if loadSize == 1024:
            encoder += [nn.Conv2d(input_nc, ngf, kernel_size=[3,3], stride=1), norm_layer(ngf), activation]
            encoder += [nn.MaxPool2d(kernel_size=[3,3], stride=2)]
            encoder += [nn.Conv2d(ngf, ngf, kernel_size=[3,3], stride=1), norm_layer(ngf), activation]
            encoder += [nn.MaxPool2d(kernel_size=[3,3], stride=2)]
        # Conv1
        else:
            encoder += [nn.Conv2d(input_nc, ngf, kernel_size=[3,3], stride=1), norm_layer(ngf), activation]
            encoder += [nn.MaxPool2d(kernel_size=[3,3], stride=2)]
        # Conv2
        encoder += [nn.Conv2d(ngf, ngf*2, kernel_size=[3,3], stride=1), norm_layer(ngf*2), activation]
        encoder += [nn.MaxPool2d(kernel_size=[3,3], stride=2)]
        # Conv3
        encoder += [nn.Conv2d(ngf*2, ngf*4, kernel_size=[3,3], stride=1), norm_layer(ngf*4), activation]
        encoder += [nn.Conv2d(ngf*4, ngf*4, kernel_size=[3,3], stride=1), norm_layer(ngf*4), activation]
        encoder += [nn.MaxPool2d(kernel_size=[3,3], stride=2)]
        # Conv4
        encoder += [nn.Conv2d(ngf*4, ngf*8, kernel_size=[3,3], stride=1), norm_layer(ngf*8), activation]
        encoder += [nn.Conv2d(ngf*8, ngf*8, kernel_size=[3,3], stride=1), norm_layer(ngf*8), activation]
        encoder += [nn.MaxPool2d(kernel_size=[3,3], stride=2)]

        # Conv5
        encoder += [nn.Conv2d(ngf*8, 500, kernel_size=[3,3], stride=1), norm_layer(500), activation]
        encoder += [nn.Conv2d(500,   500, kernel_size=[3,3], stride=1), norm_layer(500), activation]
        if dataset == 'cityscapes':
            encoder += [nn.AvgPool2d(kernel_size=[8, 16], stride=[8, 16])]
        else:
            encoder += [nn.AvgPool2d(kernel_size=[8, 26], stride=[8, 26])]

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_0 = nn.Sequential(
            nn.Linear(100, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc_0[2].weight.data.zero_()
        self.fc_loc_0[2].bias.data.copy_(torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float))

        self.fc_loc_1 = copy.deepcopy(self.fc_loc_0)
        self.fc_loc_2 = copy.deepcopy(self.fc_loc_0)
        self.fc_loc_3 = copy.deepcopy(self.fc_loc_0)
        self.fc_loc_4 = copy.deepcopy(self.fc_loc_0)
        self.encoder = nn.Sequential(*encoder)


    def theta2affine(self, theta, loadSize):
        
        #print("theta =", theta.size())
        bs = theta.size()[0]
        size = torch.empty(bs, 2, 3)
        #print(size.type())
        affine_matrix = torch.zeros_like(size).cuda()
        #print(affine_matrix.type())
        sx, sy = theta[:,  0], theta[:, 1]
        rotation, shear = theta[:, 2], theta[:, 3]
        tx, ty = theta[:, 4], theta[:, 5]
        sx = sx.clamp(min=0.6, max=1.4)
        sy = sy.clamp(min=0.6, max=1.4)
        #if loadSize == 1024:
        #    tx*=2
        #    ty*=2
        # sx sy rotation shear tx ty
        affine_matrix[:, 0, 0] = sx * torch.cos(rotation)
        affine_matrix[:, 0, 1] = - sy * torch.sin(rotation + shear)
        affine_matrix[:, 0, 2] = tx
        affine_matrix[:, 1, 0] = sx * torch.sin(rotation)
        affine_matrix[:, 1, 1] = sy * torch.cos(rotation + shear)
        affine_matrix[:, 1, 2] = ty
        affine_matrix = affine_matrix.view(-1, 2, 3)
        return affine_matrix

    def clip_mask(self, mask):
        one_ = torch.ones_like(mask)
        zero_ = torch.zeros_like(mask)
        return torch.where(mask > 0.5, one_, zero_)

    def warp(self, affine_matrix, x, mask):
        grid = F.affine_grid(affine_matrix, x.size())
        t_x = F.grid_sample(x, grid)
        mask_grid = F.affine_grid(affine_matrix, mask.size())
        t_mask = F.grid_sample(mask, mask_grid)
        t_mask = self.clip_mask(t_mask)
        return t_x, t_mask

    # Spatial transformer network forward function
    def stn(self, x, mask, feature, loadSize):
        params = self.encoder(feature)
        #print("params = ", params.size())
        params = params.view(-1, 500)
        #print("params = ", params.size())
        # STN for object timestamp t + 1
        trans_x = []
        trans_mask = []
        affine_matrix = []
        theta_0 = self.fc_loc_0(params[:,:100])
        theta_0 = theta_0.view([-1, 6])
        affine_0 = self.theta2affine(theta_0, loadSize)
        x_0, mask_0 = self.warp(affine_0, x, mask)
        trans_x.append(x_0)
        trans_mask.append(mask_0)
        affine_matrix.append(theta_0)

        theta_1 = self.fc_loc_1(params[:,100:200])
        theta_1 = theta_1.view([-1, 6])
        affine_1 = self.theta2affine(theta_1, loadSize)
        x_1, mask_1 = self.warp(affine_1, x, mask)
        trans_x.append(x_1)
        trans_mask.append(mask_1)
        affine_matrix.append(theta_1)

        theta_2 = self.fc_loc_2(params[:,200:300])
        theta_2 = theta_2.view([-1, 6])
        affine_2 = self.theta2affine(theta_2, loadSize)
        x_2, mask_2 = self.warp(affine_2, x, mask)
        trans_x.append(x_2)
        trans_mask.append(mask_2)
        affine_matrix.append(theta_2)

        theta_3 = self.fc_loc_3(params[:,300:400])
        theta_3 = theta_3.view([-1, 6])
        affine_3 = self.theta2affine(theta_3, loadSize)
        x_3, mask_3 = self.warp(affine_3, x, mask)
        trans_x.append(x_3)
        trans_mask.append(mask_3)
        affine_matrix.append(theta_3)
                
        theta_4 = self.fc_loc_4(params[:,400:])
        theta_4 = theta_4.view([-1, 6])
        affine_4 = self.theta2affine(theta_4, loadSize)
        x_4, mask_4 = self.warp(affine_4, x, mask)
        trans_x.append(x_4)
        trans_mask.append(mask_4)
        affine_matrix.append(theta_4)
        
        return trans_x, trans_mask, affine_matrix

    def forward(self, loadSize, combine_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, mask_reshaped, target_back_reshaped, last_object=None, last_mask=None):
        input = torch.cat([combine_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, mask_reshaped, target_back_reshaped], dim=1)
        if last_object is not None:
            last_object = last_object
            last_mask = last_mask
        else:
            last_object = combine_reshaped[:, -3:, :, :] * mask_reshaped[:,-1:,:,:].repeat(1,3,1,1)
            last_mask = mask_reshaped[:, -1:, ...]
        #print("input size = ", input.size())
        warped_x, warped_mask, affine_matrix = self.stn(last_object, last_mask, input, loadSize)
        pred_complete = []
        for i in range(self.tOut):
            cnt_pred_mask = warped_mask[i].repeat(1,3,1,1)
            #if loadSize == 1024:
            #    cnt_pred_complete = target_back_map[:, i, ...] * (1.0 - cnt_pred_mask) + warped_x[i] * cnt_pred_mask
            #else:
            cnt_pred_complete = target_back_reshaped[:, i*3:(i+1)*3, ...] * (1.0 - cnt_pred_mask) + warped_x[i] * cnt_pred_mask
            pred_complete.append(cnt_pred_complete)
        return warped_x, warped_mask, affine_matrix, pred_complete


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
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf * (2 ** (num_D - 1 - i))), n_layers, norm_layer,
                                       getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

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
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
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
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
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
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, mask):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            _, ch, h, w = x_vgg[i].size()
            curr_mask = F.upsample_nearest(mask, (h, w))
            curr_mask = curr_mask.expand(-1, ch, -1, -1)
            loss += self.weights[i] * self.criterion(x_vgg[i] * curr_mask, y_vgg[i].detach() * curr_mask)
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, label_nc):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss2d()

    def forward(self, output, label):
        label = label.long().max(1)[1]
        output = self.softmax(output)
        return self.criterion(output, label)


class MaskOneL1Loss(nn.Module):
    def __init__(self):
        super(MaskOneL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, input, target, mask):
        loss = self.criterion(input * mask, target * mask) / (torch.sum(mask) + 1.0)
        return loss


class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        # self.weights = [0.5, 1, 2, 8, 32]
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
            if i != len(self.weights) - 1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss


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
