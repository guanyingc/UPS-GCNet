import torch
import torch.nn as nn
from models import model_utils

class FeatExtractor(nn.Module):
    def __init__(self, opt, c_in=6, c_out=128):
        super(FeatExtractor, self).__init__()
        batchNorm = opt['use_BN']
        self.conv1 = model_utils.conv_layer(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv_layer(batchNorm, 64,   128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv_layer(batchNorm, 128,  128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv_layer(batchNorm, 128,  256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv_layer(batchNorm, 256,  256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv_layer(256, 128)
        self.conv7 = model_utils.conv_layer(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat   = out_feat.view(-1)
        return out_feat, [n, c, h, w]

class Regressor(nn.Module):
    def __init__(self, opt, c_in): 
        super(Regressor, self).__init__()
        batchNorm = opt['use_BN']
        self.deconv1 = model_utils.conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv_layer(128, 64)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class PS_FCN(nn.Module):
    def __init__(self, opt, c_in):
        super(PS_FCN, self).__init__()
        self.opt = opt
        self.extractor = FeatExtractor(self.opt, c_in=c_in, c_out=128)
        self.regressor = Regressor(self.opt, c_in=128)

    def forward(self, inputs):
        feats = torch.Tensor()
        for i in range(len(inputs)):
            feat, shape = self.extractor(inputs[i])
            if i == 0:
                feats = feat
            else:
                if self.opt['fuse_type'] == 'mean':
                    feats = torch.stack([feats, feat], 1).sum(1)
                else:
                    feats, _ = torch.stack([feats, feat], 1).max(1)
        if self.opt['fuse_type'] == 'mean':
            feats = feats / len(img_split)
        feat_fused = feats
        normal = self.regressor(feat_fused, shape)
        pred = {}
        pred['normal'] = normal
        return pred
