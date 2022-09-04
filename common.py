import math, torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = 0.2
        theta = 0.8
        
        dual_block = [
            nn.Sequential(
                # nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                # nn.LeakyReLU(negative_slope=negval, inplace=True)
                Conv2d_Ada_Cross(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False, theta= theta),
                nn.BatchNorm2d(nFeat),
                nn.LeakyReLU(negative_slope=negval, inplace=True)   
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    # nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    # nn.LeakyReLU(negative_slope=negval, inplace=True)
                    Conv2d_Ada_Cross(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False, theta= theta),
                    nn.BatchNorm2d(nFeat),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)   
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Conv2d_Ada_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Ada_Cross, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.norm = nn.InstanceNorm2d(in_channels//2, affine=True)
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)

    def top_k(self, input, k=5):
        input = input.data
        input_r1 = input.view(input.size()[0]*input.size()[1],9)
        _, b = torch.topk(input_r1[:,0:9], k, 1, largest= True) 
        c = torch.zeros(input_r1.size()[0], input_r1.size()[1], dtype=torch.int)
        # print(c.shape)
        for i in range((c.size()[0])):
            c[i, b[i,:]] = 1
        out = c.cuda() * input_r1
        out = out.view(input.size()[0]*input.size()[1],3,3)
        return out

    def forward(self, x):
        
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:,:,:,0], self.conv.weight[:,:,:,1], self.conv.weight[:,:,:,2], self.conv.weight[:,:,:,3], self.conv.weight[:,:,:,4], self.conv.weight[:,:,:,5], self.conv.weight[:,:,:,6], self.conv.weight[:,:,:,7], self.conv.weight[:,:,:,8]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        index = self.top_k(conv_weight)

        #x = self.conv_1(x)
        #out_1, out_2 = torch.chunk(x, 2, dim=1)
        #x = torch.cat([self.norm(out_1), out_2], dim=1)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]

            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:,:,:,0], tensor_zeros, self.conv.weight[:,:,:,1], self.conv.weight[:,:,:,2], self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        
        
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:,:,:,0], tensor_zeros, self.conv.weight[:,:,:,1], tensor_zeros, self.conv.weight[:,:,:,2], tensor_zeros, self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Cross, self).__init__() 
        self.conv_hv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_135 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 6), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_45 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 6), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta1 = theta
        self.theta2 = theta
        self.theta3 = theta
        self.theta4 = theta
        self.out_conv = default_conv(out_channels*4, out_channels, 1)

    def forward(self, x):
        
        [C_out,C_in,H_k,W_k] = self.conv_hv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()

        ## Hori_Veri
        conv_weight = torch.cat((tensor_zeros, self.conv_hv.weight[:,:,:,0], tensor_zeros, self.conv_hv.weight[:,:,:,1], self.conv_hv.weight[:,:,:,2], self.conv_hv.weight[:,:,:,3], tensor_zeros, self.conv_hv.weight[:,:,:,4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv_hv.bias, stride=self.conv_hv.stride, padding=self.conv_hv.padding)
        [C_out,C_in, kernel_size,kernel_size] = self.conv_hv.weight.shape
        kernel_diff = self.conv_hv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv_hv.bias, stride=self.conv_hv.stride, padding=0, groups=self.conv_hv.groups)
        out_Hori_Veri = out_normal - self.theta1 * out_diff

        ## Diag
        conv_weight = torch.cat((self.conv_d.weight[:,:,:,0], tensor_zeros, self.conv_d.weight[:,:,:,1], tensor_zeros, self.conv_d.weight[:,:,:,2], tensor_zeros, self.conv_d.weight[:,:,:,3], tensor_zeros, self.conv_d.weight[:,:,:,4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv_d.bias, stride=self.conv_d.stride, padding=self.conv_d.padding)
        [C_out,C_in, kernel_size,kernel_size] = self.conv_d.weight.shape
        kernel_diff = self.conv_d.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv_d.bias, stride=self.conv_d.stride, padding=0, groups=self.conv_d.groups)
        out_Diag = out_normal - self.theta2 * out_diff

        ## 135
        conv_weight = torch.cat((self.conv_135.weight[:,:,:,0], self.conv_135.weight[:,:,:,1], tensor_zeros, self.conv_135.weight[:,:,:,2], tensor_zeros, self.conv_135.weight[:,:,:,3], tensor_zeros, self.conv_135.weight[:,:,:,4], self.conv_135.weight[:,:,:,5]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv_135.bias, stride=self.conv_135.stride, padding=self.conv_135.padding)
        [C_out,C_in, kernel_size,kernel_size] = self.conv_135.weight.shape
        kernel_diff = self.conv_135.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv_135.bias, stride=self.conv_135.stride, padding=0, groups=self.conv_135.groups)
        out_135 = out_normal - self.theta3 * out_diff


        ## 45
        conv_weight = torch.cat((tensor_zeros, self.conv_45.weight[:,:,:,0], self.conv_45.weight[:,:,:,1], self.conv_45.weight[:,:,:,2], tensor_zeros, self.conv_45.weight[:,:,:,3], self.conv_45.weight[:,:,:,4], self.conv_45.weight[:,:,:,5], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv_45.bias, stride=self.conv_45.stride, padding=self.conv_45.padding)
        [C_out,C_in, kernel_size,kernel_size] = self.conv_45.weight.shape
        kernel_diff = self.conv_45.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv_45.bias, stride=self.conv_45.stride, padding=0, groups=self.conv_45.groups)
        out_45 = out_normal - self.theta4 * out_diff

        out_all = torch.cat((out_Hori_Veri, out_Diag, out_135, out_45), 1)
        out = self.out_conv(out_all)
        # out = 0.25 * out_Hori_Veri + 0.25 * out_Diag + 0.25 * out_135 + 0.25 * out_45 
        return out
