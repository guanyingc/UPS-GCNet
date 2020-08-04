import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict
from utils import eval_utils

class DirectionCrit(object):
    def __init__(self, opt, log):
        log.print_write('==> [Criterion] Using DirectionCrit')
        self.dirs_x_crit = torch.nn.CrossEntropyLoss()
        self.dirs_y_crit = torch.nn.CrossEntropyLoss()
        self.opt = opt

    def __call__(self, est_dirs_x, est_dirs_y, gt_dirs):
        loss = 0
        est_dir_x = est_dirs_x.view(-1, self.opt['dirs_cls'])
        est_dir_y = est_dirs_y.view(-1, self.opt['dirs_cls'])
        gt_dir_x, gt_dir_y = eval_utils.spherical_dirs_to_class(gt_dirs.view(-1, 3), self.opt['dirs_cls'])
    
        dirs_x_loss = self.dirs_x_crit(est_dir_x, gt_dir_x)
        dirs_y_loss = self.dirs_y_crit(est_dir_y, gt_dir_y)
        loss_term = {}
        loss_term['D_x_loss'] = dirs_x_loss.item()
        loss_term['D_y_loss'] = dirs_y_loss.item()
        loss = dirs_x_loss + dirs_y_loss
        return loss, loss_term

class IntensityCrit(object):
    def __init__(self, opt, log):
        log.print_write('==> [Criterion] Using IntensityCrit')
        self.ints_crit = torch.nn.CrossEntropyLoss()
        self.opt = opt

    def __call__(self, est_ints, gt_ints):
        n, c = gt_ints.shape[:2]
        est_intens = est_ints.view(-1, self.opt['ints_cls'])
        gt_ints = gt_ints[:,:,0].view(-1, 1)
        gt_int_class = eval_utils.light_ints_to_class(gt_ints, self.opt['ints_cls'])
        ints_loss = self.ints_crit(est_intens, gt_int_class)
        loss_term = {}
        loss_term['I_loss'] = ints_loss.item()
        return ints_loss, loss_term

class NormalCrit(object):
    def __init__(self, opt, log):
        log.print_write('==> [Criterion] Using IntensityCrit')
        self.normal_crit = torch.nn.CosineEmbeddingLoss()
        self.opt = opt

    def __call__(self, est_normal, gt_normal):
        n_est, n_tar = est_normal, gt_normal
        n_num = n_tar.nelement() // n_tar.shape[1]
        if not hasattr(self, 'n_flag') or n_num != self.n_flag.nelement():
            self.n_flag = n_tar.data.new().resize_(n_num).fill_(1)
        self.out_reshape = n_est.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        self.gt_reshape  = n_tar.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        normal_loss = self.normal_crit(self.out_reshape, self.gt_reshape, self.n_flag)
        loss_term = {'N_loss': normal_loss.item()}
        return normal_loss, loss_term

def get_lr_scheduler(opt, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=opt['milestones'], gamma=opt['lr_decay'], last_epoch=opt['start_epoch']-2)
    return scheduler

def init_weights(net, init_type='kaiming'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func) 

def init_net(net, init_type='kaiming', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type)
    return net

def get_params_num(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

## Common Network Blocks
def activation(afunc='LReLU'):
    if afunc == 'LReLU':
        return nn.LeakyReLU(0.1, inplace=True)
    elif afunc == 'ReLU':
        return nn.ReLU(inplace=True)
    else:
        raise Exception('Unknown activation function')

def conv_layer(batchNorm, cin, cout, k=3, stride=1, pad=-1, afunc='LReLU'):
    if type(pad) != tuple:
        pad = pad if pad >= 0 else (k - 1) // 2
    mList = [nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True)]
    if batchNorm: 
        print('=> convolutional layer with bachnorm')
        mList.append(nn.BatchNorm2d(cout))
    mList.append(activation(afunc))
    return nn.Sequential(*mList)

def deconv_layer(cin, cout, batchNorm=False, afunc='LReLU'):
    mList = [nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False)]
    if batchNorm: 
        print('=> deconvolutional layer with bachnorm')
        mList.append(nn.BatchNorm2d(cout))
    mList.append(activation(afunc))
    return nn.Sequential(*mList)

def upconv_layer(cin, cout, batchNorm=False, afunc='LReLU'):
    mList = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
             nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=True)]
    if batchNorm: 
        print('=> deconvolutional layer with bachnorm')
        mList.append(nn.BatchNorm2d(cout))
    mList.append(activation(afunc))
    return nn.Sequential(*mList)

def output_conv(cin, cout, k=1, stride=1, pad=0):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True))

def fc_layer(cin, cout):
    return nn.Sequential(
            nn.Linear(cin, cout),
            nn.LeakyReLU(0.1, inplace=True))
