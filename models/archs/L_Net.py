import torch
import torch.nn as nn
from models import model_utils
from utils import eval_utils
from collections import OrderedDict
import numpy as np

def fuse_features(feats, opt):
    if opt['fuse_type'] == 'mean':
        feat_fused = torch.stack(feats, 1).mean(1)
    elif opt['fuse_type'] == 'max':
        feat_fused, _ = torch.stack(feats, 1).max(1)
    return feat_fused

def spherical_class_to_dirs(x_cls, y_cls, cls_num):
    theta = (x_cls.float() + 0.5) / cls_num * 180 - 90
    phi   = (y_cls.float() + 0.5) / cls_num * 180 - 90
    theta = theta.clamp(-90, 90) / 180.0 * np.pi
    phi   = phi.clamp(-90, 90) / 180.0 * np.pi

    tan2_theta = pow(torch.tan(theta), 2)
    y = torch.sin(phi)
    z = torch.sqrt((1 - y * y) / (1 + tan2_theta))
    x = z * torch.tan(theta)
    dirs = torch.stack([x,y,z], 1)
    dirs = dirs / dirs.norm(p=2, dim=1, keepdim=True)
    return dirs

def convert_dirs(l_dirs_x, l_dirs_y, opt, dirs_step):
    # soft-argmax
    dirs_x = torch.cat(l_dirs_x, 0).squeeze()
    dirs_y = torch.cat(l_dirs_y, 0).squeeze()
    x_prob = torch.nn.functional.softmax(dirs_x, dim=1)
    y_prob = torch.nn.functional.softmax(dirs_y, dim=1)
    x_idx = (x_prob * dirs_step).sum(1)
    y_idx = (y_prob * dirs_step).sum(1)
    dirs = spherical_class_to_dirs(x_idx, y_idx, opt['dirs_cls'])
    return dirs

def convert_intens(l_ints, opt, ints_step):
    # soft-argmax
    l_ints = torch.cat(l_ints, 0).view(-1, opt['ints_cls'])
    int_prob = torch.nn.functional.softmax(l_ints, dim=1)
    idx = (int_prob * ints_step).sum(1)
    ints = eval_utils.class_to_light_ints(idx, opt['ints_cls'])
    ints = ints.view(-1, 1).repeat(1, 3)
    return ints

# Classification
class FeatExtractor(nn.Module):
    def __init__(self, opt, c_in=4, c_out=256):
        super(FeatExtractor, self).__init__()
        batchNorm = opt['use_BN']
        self.conv1 = model_utils.conv_layer(batchNorm, c_in, 32,  k=3, stride=2, pad=1, afunc='LReLU')
        self.conv2 = model_utils.conv_layer(batchNorm, 32,   64,  k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv_layer(batchNorm, 64,   64,  k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv_layer(batchNorm, 64,  128,  k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv_layer(batchNorm, 128, 128,  k=3, stride=2, pad=1)
        self.conv7 = model_utils.conv_layer(batchNorm, 128, 256,  k=3, stride=1, pad=1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        return out

class Classifier(nn.Module):
    def __init__(self, opt, c_in):
        super(Classifier, self).__init__()
        batchNorm = opt['use_BN']
        self.conv1 = model_utils.conv_layer(batchNorm, 512,  128, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv_layer(batchNorm, 128,  128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv_layer(batchNorm, 128,  128, k=3, stride=2, pad=1)
        self.conv4 = model_utils.conv_layer(batchNorm, 128,  128, k=3, stride=2, pad=1)
        self.opt = opt
        
        self.dir_x_est = nn.Sequential(
                    model_utils.conv_layer(batchNorm, 128, 64,  k=1, stride=1, pad=0),
                    model_utils.output_conv(64, opt['dirs_cls'], k=1, stride=1, pad=0))

        self.dir_y_est = nn.Sequential(
                    model_utils.conv_layer(batchNorm, 128, 64,  k=1, stride=1, pad=0),
                    model_utils.output_conv(64, opt['dirs_cls'], k=1, stride=1, pad=0))

        self.int_est = nn.Sequential(
                    model_utils.conv_layer(batchNorm, 128, 64,  k=1, stride=1, pad=0),
                    model_utils.output_conv(64, opt['ints_cls'], k=1, stride=1, pad=0))

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        outputs = {}
        outputs['dir_x'] = self.dir_x_est(out)
        outputs['dir_y'] = self.dir_y_est(out)
        outputs['ints'] = self.int_est(out)
        return outputs

class L_Net(nn.Module):
    def __init__(self, opt, c_in):
        super(L_Net, self).__init__()
        self.opt = opt
        self.featExtractor = FeatExtractor(self.opt, c_in=c_in, c_out=256)
        self.classifier = Classifier(self.opt, c_in=512)

        d_cls, i_cls = self.opt['dirs_cls'], self.opt['ints_cls']
        self.register_buffer('dirs_step', torch.linspace(0, d_cls-1, d_cls))
        self.register_buffer('ints_step', torch.linspace(0, i_cls-1, i_cls))

    def forward(self, inputs):
        feats = []
        for i in range(len(inputs)):
            out_feat = self.featExtractor(inputs[i])
            feats.append(out_feat)
        feat_fused = fuse_features(feats, self.opt)

        l_dirs_x, l_dirs_y, l_ints = [], [], []
        for i in range(len(inputs)):
            net_input = torch.cat([feats[i], feat_fused], 1)
            outputs = self.classifier(net_input)
            l_dirs_x.append(outputs['dir_x'])
            l_dirs_y.append(outputs['dir_y'])
            l_ints.append(outputs['ints'])

        pred = OrderedDict()
        batch = inputs[0].shape[0]

        dirs = convert_dirs(l_dirs_x, l_dirs_y, self.opt, self.dirs_step)
        pred['dirs'] = torch.stack(torch.split(dirs, batch, 0), 1)
        pred['dirs_x'] = torch.stack(l_dirs_x, 1).view(batch, len(inputs), self.opt['dirs_cls'])
        pred['dirs_y'] = torch.stack(l_dirs_y, 1).view(batch, len(inputs), self.opt['dirs_cls'])

        intens = convert_intens(l_ints, self.opt, self.ints_step)
        pred['intens'] = torch.stack(torch.split(intens, batch, 0), 1)
        pred['ints'] = torch.stack(l_ints, 1).view(batch, len(inputs), self.opt['ints_cls'])
        return pred
