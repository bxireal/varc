from __future__ import print_function
import argparse
import os
from math import log10
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ldc import CelebADataset
# from networks import define_G, define_D, GANLoss, print_network
from networks import _NetBFSR, print_network, _NetD, GANLoss
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F
import archs.EDVR_arch as EDVR_arch
# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', default="data", help='dataset')
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning Rate. Default=0.0001')
parser.add_argument("--declr", type=float, default=0.5, help="Decreasing Forward Net Learning Rate. Default=0.5")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument('--cuda', default="true", action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument("--resume", default="/data/JXR/VSRC/checkpoint/data/netG_model_epoch_4.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")

torch.cuda.set_device(3)
# gpus_list = [2, 3]
# global opt
#     opt = parser.parse_args()
#     print(opt)
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

# class MYSampler(data.Sampler):
#     r"""Samples elements randomly from a given list of indices, without replacement.
#     Arguments:
#         indices (sequence): a sequence of indices
#     """
#
#     def __init__(self, indices):
#         self.indices = indices
#
#     def __iter__(self):
#         return (self.indices[i] for i in range(len(self.indices)))
#
#     def __len__(self):
#         return len(self.indices)

# newlist = [i for i in range(70000)]
# # print(newlist)
# train_sampler = MYSampler(indices=newlist)



print('===> Loading datasets')
root_path = "/data/JXR/VSRC/dataset/data/"
train_set = get_training_set(root_path)
# test_set = get_test_set(root_path)
training_data_loader = DataLoader(dataset=train_set,  num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False, pin_memory = True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


print('===> Building model')
# netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, [0])
netG = EDVR_arch.EDVR(nf = 64, nframes=7, groups =8, front_RBs=5 ,back_RBs=10 ,center= None,predeblur=True , HR_in= False, w_TSA= True)
netD = _NetD()
# netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'batch', False, [0])

criterionGAN = GANLoss()
# criterionL11 = nn.L1Loss(size_average=False)
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
criterionJC = nn.CrossEntropyLoss()

# setup optimizer
# optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
# print_network(netG)
# print_network(netD)
print('-----------------------------------------------')

# real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 128, 128)
# real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 128, 128)
# attr = torch.FloatTensor(opt.batchSize, 40, 1, 1)

if opt.cuda:
    netD = netD.cuda(3)
    # model = netG
    # model = torch.nn.DataParallel(model, device_ids=[2, 3])
    model = netG.cuda(3)
    criterionGAN = criterionGAN.cuda(3)
    criterionL1 = criterionL1.cuda(3)
    # criterionMSE = criterionMSE.cuda()
    # real_a = real_a.cuda()
    # real_b = real_b.cuda()
    # attr = attr.cuda()

# real_a = Variable(real_a)
# real_b = Variable(real_b)


# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        epoch = checkpoint["epoch"] + 1
        netG.load_state_dict(checkpoint["netG"].state_dict())
        # netG.load_state_dict(checkpoint.state_dict())
        # opt.lr = opt.lr*4
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def adjust_learning_rate(optimizer, epoch, lr_now, epoch_step, declr_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = lr_now * (declr_rate ** (epoch // epoch_step))
    return lr


def train(epoch):

    # for iteration, batch in enumerate(training_data_loader, 1):
    for iteration, batch in enumerate(training_data_loader):
    # for batch in training_data_loader:




        lr = adjust_learning_rate(optimizerG, epoch - 1, opt.lr, opt.step, opt.declr)

        for param_group in optimizerG.param_groups:
            param_group["lr"] = lr
        print("epoch =", epoch, "lr_forward =", optimizerG.param_groups[0]["lr"])
        # forward
        # input_p3, input_p2, input_p1, input, input_n1, input_n2, input_n3, gt = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]
        #
        # input_p3 = input_p3.cuda(2)
        # input_p2 = input_p2.cuda(2)
        # input_p1 = input_p1.cuda(2)
        # input = input.cuda(2)
        # input_n1 = input_n1.cuda(2)
        # input_n2 = input_n2.cuda(2)
        # input_n3 = input_n3.cuda(2)
        # gt = gt.cuda(2)
        input , gt = batch[0] , batch[1]
        input = input.cuda(3)
        gt = gt.cuda(3)


        # sr = netG(input_p2, input_p1, input, input_n1, input_n2)

        # sr_p1 = model(F.interpolate(input_p2, scale_factor=4, mode='bilinear'), input_p1, F.interpolate(input, scale_factor=4, mode='bilinear'))
        # sr_n1 = model(F.interpolate(input, scale_factor=4, mode='bilinear'), input_n1, F.interpolate(input_n2, scale_factor=4, mode='bilinear'))
        # sr = model(sr_p1, input, sr_n1)
        sr = model(input)
        # L3_offset = F.interpolate(input_p2, scale_factor=4, mode='bilinear')

        # ############################
        # # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        # ###########################
        #
        optimizerD.zero_grad()
        #
        # # train with fake
        # # fake_ab = torch.cat((real_a, fake_b), 1)
        # # pred_fake = netD.forward(fake_ab.detach())
        pred_fake, aa = netD(sr)
        loss_d_fake = criterionGAN(pred_fake, False)
        #
        # # train with real
        # # real_ab = torch.cat((real_a, real_b), 1)
        pred_real, aa = netD(gt)
        loss_d_real = criterionGAN(pred_real, True)
        #
        # # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward(retain_graph=True)

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()

        # # fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake, con1 = netD(sr)
        _, con2 = netD(gt)
        loss_con = criterionL1(con1.detach(), con2.detach())
        loss_g_gan = criterionGAN(pred_fake, True)

    # optimizerG.zero_grad()

        loss_g = criterionL1(sr, gt)

        loss = loss_g + 0.1 * loss_g_gan + 0.1 * loss_con


        loss.backward()

        optimizerG.step()

        # print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} ".format(
        # epoch, iteration, len(training_data_loader), loss_g.data))

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_D: {:.4f} Loss_G: {:.4f}".format(
           epoch, iteration, len(training_data_loader), loss_g.data, loss_g_gan.data, loss_con))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint_1", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)

    state = {"epoch": epoch, "netG": netG}
    torch.save(state, net_g_model_out_path)
    ## net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
    ## torch.save(netG, net_g_model_out_path)
    ## # torch.save(netD, net_d_model_out_path)
    print("Checkpoint saved to {}".format("model/"))


for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    if epoch % 1 == 0:
        checkpoint(epoch)
