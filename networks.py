import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64*7, out_channels=64*7, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64*7, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64*7, out_channels=64*7, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64*7, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output


class Res2Block(nn.Module):

    def __init__(self, n_feats=64, num=4, width=16):
        super(Res2Block, self).__init__()
        self.num = num
        self.width = width

        self.relu = nn.LeakyReLU(0.2)
        self.top = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
        )

        convs = []
        for i in range(self.num):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False))
        self.convs = nn.ModuleList(convs)
        self.end = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):

        out = self.top(x)

        spx = torch.split(out, self.width, 1)
        for i in range(self.num):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.end(out)

        return out

class Predeblur_ResNet_Pyramid(nn.Module):
    def __init__(self, nf=64, HR_in=False):
        '''
        HR_in: True if the inputs are high spatial size
        '''

        super(Predeblur_ResNet_Pyramid, self).__init__()
        self.HR_in = True if HR_in else False
        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.RB_L1_1 = Res2Block()
        self.RB_L1_2 = Res2Block()
        self.RB_L1_3 = Res2Block()
        self.RB_L1_4 = Res2Block()
        self.RB_L1_5 = Res2Block()
        self.RB_L2_1 = Res2Block()
        self.RB_L2_2 = Res2Block()
        self.RB_L3_1 = Res2Block()
        self.deblur_L2_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.deblur_L3_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.lrelu = nn.PReLU()

    def forward(self, x):
        if self.HR_in:
            L1_fea = self.lrelu(self.conv_first_1(x))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = F.interpolate(self.RB_L3_1(L3_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = F.interpolate(self.RB_L2_2(L2_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out


class Fefusion(nn.Module):
    def __init__(self, nf=64):

        super(Fefusion, self).__init__()

        self.conv = nn.Conv2d(nf*7, nf*7, 3, 1, 1, bias=True)

        self.R2B1 = Res2Block()
        self.R2B2 = Res2Block()
        self.R2B3 = Res2Block()

        self.midm1 = _Residual_Block()
        self.midm2 = _Residual_Block()
        self.midm3 = _Residual_Block()
        self.midmm1 = _Residual_Block()
        self.midmm2 = _Residual_Block()

        self.down_conv1 = nn.Conv2d(nf*7, nf*7, 3, 2, 1, bias=True)
        self.down_conv2 = nn.Conv2d(nf*7, nf*7, 3, 2, 1, bias=True)

        self.fuse_conv = nn.Conv2d(nf*7, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.PReLU()

    def forward(self, x):

        L1_fea = self.lrelu(self.conv(x))
        L2_fea = self.lrelu(self.down_conv1(L1_fea))
        L3_fea = self.lrelu(self.down_conv2(L2_fea))
        L3_fea = F.interpolate(self.midmm1(L3_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L2_fea = self.midm1(L2_fea) + L3_fea
        L2_fea = F.interpolate(self.midmm2(L2_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L1_fea = self.midm3(self.midm2(L1_fea)) + L2_fea
        L1_fea = self.lrelu(self.fuse_conv(L1_fea))
        out = self.R2B3(self.R2B2(self.R2B1(L1_fea)))
        return out

# class Fefusion(nn.Module):
#     def __init__(self, nf=64):
#
#         super(Fefusion, self).__init__()
#
#
#         self.R2B1 = Res2Block()
#         self.R2B2 = Res2Block()
#         self.R2B3 = Res2Block()
#
#
#         self.fuse_conv = nn.Conv2d(nf*7, nf, 3, 1, 1, bias=True)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def forward(self, x):
#
#         L1_fea = self.lrelu(self.fuse_conv(x))
#         out = self.R2B3(self.R2B2(self.R2B1(L1_fea)))
#         return out




class MC_Block(nn.Module):
    def __init__(self):
        super(MC_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.in4 = nn.InstanceNorm2d(256, affine=True)
        self.relu = nn.PReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in5 = nn.InstanceNorm2d(64, affine=True)
        self.sigmoid = nn.Sigmoid()
        self.up2 = nn.PixelShuffle(2)


    def forward(self, f1, f2):
        identity_data = self.sigmoid(f1 - f2)
        jxr1 = self.relu(self.in1(self.conv1(identity_data)))
        jxr2 = self.relu(self.in2(self.conv2(jxr1)))
        jxr3 = self.up2(self.relu(self.in3(self.conv3(jxr2))))
        jxr3 = jxr3 + jxr1
        jxr4 = self.up2(self.relu(self.in4(self.conv4(jxr3))))
        jxr4 = jxr4 + identity_data
        output = self.in5(self.conv5(jxr4))
        output = f1 + output * identity_data

        return output


class _NetBFSR(nn.Module):
    def __init__(self):
        super(_NetBFSR, self).__init__()
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.relu = nn.PReLU()

        self.pre_deblur = Predeblur_ResNet_Pyramid()
        self.conv_1 = nn.Conv2d(3, 64, 3, 1, 1,bias=False)
        self.mc_block = MC_Block()
        self.fefusion = Fefusion()

        self.residual_pre = self.make_layer(Res2Block, 10)
        self.residual = self.make_layer(Res2Block, 10)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        self.downscale4x = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


    def forward(self, input_p3, input_p2, input_p1, input, input_n1, input_n2, input_n3):
        input_p3 = self.conv_1(input_p3)
        input_p2 = self.conv_1(input_p2)
        input_p1 = self.conv_1(input_p1)
        input = self.conv_1(input)
        input_n1 = self.conv_1(input_n1)
        input_n2 = self.conv_1(input_n2)
        input_n3 = self.conv_1(input_n3)


        x = torch.cat((input_p3, input_p2, input_p1, input, input_n1, input_n2, input_n3), 1)
        # out = self.relu(self.conv_input(x))
        out = self.fefusion(x)

        out = self.residual_pre(out)
        out = self.pre_deblur(out)

        # mcs = []
        # a, b, c, d = out.size()
        # batch = int(a/7)
        # for i in range(7):
        #    mc = self.mc_block(out[i*batch:(i+1)*batch, :, :, :], out[3*batch:(3+1)*batch, :, :, :])
        #    mcs.append(mc)
        # out = torch.cat((mcs), 1)
        # out = self.fefusion(out)

        out = self.residual(out)


        out = self.upscale4x(out)
        out = self.conv_output(out)
        # out = out + F.interpolate(x, scale_factor=4, mode='bilinear')

        return out


class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)


        self.dfe1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.PReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.dfe2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.Tanh()
        )

        self.dfc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

        # self.conv_R = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):

        idex = self.relu(self.conv_input(x))

        a, b, _, _ = idex.size()
        fee1 = self.dfe1(idex)

        fee2 = self.dfe2(fee1).view(a,64)

        result = self.dfc(fee2)

        return result, fee1


# Defines the GAN loss which uses either LSGAN or the regular GAN.
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
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor


    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
            # target_tensor = 1
        else:
            target_tensor = self.fake_label
            # target_tensor = 0
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)





