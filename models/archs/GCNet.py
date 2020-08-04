import torch
import torch.nn as nn
from models.archs import N_Net, L_Net
from collections import OrderedDict

class GCNet(nn.Module):
    def __init__(self, opt, c_in):
        super(GCNet, self).__init__()
        self.opt = opt
        self.L_Net1 = L_Net.L_Net(opt, c_in=4)
        self.N_Net = N_Net.N_Net(opt, c_in=6)

        if self.opt['in_est_n']: 
            print('[%s]: adding estimated normal as input' % self.__class__.__name__)
            c_in += 3
        if self.opt['in_est_sd']: 
            print('[%s]: adding estimated shading as input' % self.__class__.__name__)
            c_in += 1
        self.L_Net2 = L_Net.L_Net(opt, c_in=c_in)

    def get_shading(self, normal, lights):
        batch, c, h, w = normal.shape
        normal = normal.view(batch, 3, h * w)
        shadings = []
        dirs = torch.split(lights, 1, 1)
        for l_dir in dirs:
            shading = torch.bmm(l_dir.view(batch, 1, 3), normal)
            shading = shading.clamp(0, 1).view(batch, 1, h, w)
            shadings.append(shading)
        return shadings
    
    def prepare_LNet1_inputs(self, data): 
        n, c, h, w = data['img'].shape
        t_h, t_w = self.opt['test_h'], self.opt['test_w']
        if (h == t_h and w == t_w):
            imgs = data['img'] 
        else:
            print('Rescaling images: from %dX%d to %dX%d' % (h, w, t_h, t_w))
            imgs = torch.nn.functional.interpolate(data['img'], size=(t_h, t_w), mode='bilinear', align_corners=False)
        new_data = {'img': imgs}

        inputs = list(torch.split(imgs, 3, 1))

        keys = OrderedDict({'in_mask': 'mask'})
        for k in keys:
            if not self.opt[k]: continue
            maps = data[keys[k]]
            if (maps.shape[2] != t_h or maps.shape[3] != t_w):
                maps = torch.nn.functional.interpolate(maps, size=(t_h, t_w), mode='bilinear', align_corners=False)
            if k in ['in_mask']:
                for i in range(len(inputs)):
                    inputs[i] = torch.cat([inputs[i], maps], 1)
            new_data[keys[k]] = maps
        return inputs, new_data

    def prepare_NNet_inputs(self, data, pred):
        imgs = torch.split(data['img'], 3, 1)
        dirs = torch.split(pred['dirs'], 1, 1)
        ints = torch.split(pred['intens'], 1, 1)
        
        inputs = []

        imgs_int_normalized = []
        n, c, h, w = imgs[0].shape
        for i in range(len(imgs)):
            l_int = torch.diag(1.0 / (ints[i].contiguous().view(-1) + 1e-8))
            img = imgs[i].contiguous().view(n * c, h * w)
            img = torch.mm(l_int, img).view(n, c, h, w)
            imgs_int_normalized.append(img)

        imgs = imgs_int_normalized

        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            l_dir = dirs[i] if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1)
            img_light = torch.cat([imgs[i], l_dir.expand_as(imgs[i])], 1)
            inputs.append(img_light)
        return inputs

    def prepare_LNet2_inputs(self, inputs, L_Net1_pred, N_Net_pred):
        if self.opt['in_est_n']:
            normal = N_Net_pred['normal']
            for i in range(len(inputs)):
                inputs[i] = torch.cat([inputs[i], normal], 1)
        if self.opt['in_est_sd']:
            for i in range(len(inputs)):
                inputs[i] = torch.cat([inputs[i], N_Net_pred['shading'][i]], 1)
        return inputs

    def forward(self, data):
        pred = {}
        L_Net1_inputs, new_data = self.prepare_LNet1_inputs(data)
        L_Net1_pred = self.L_Net1(L_Net1_inputs)
        for k in L_Net1_pred: 
            pred['prev_%s' % k] = L_Net1_pred[k]

        N_Net_inputs = self.prepare_NNet_inputs(new_data, L_Net1_pred)
        N_Net_pred = self.N_Net(N_Net_inputs)
        N_Net_pred['normal'] = N_Net_pred['normal'] * new_data['mask']
        for k in N_Net_pred: 
            pred['prev_%s' % k] = N_Net_pred[k]
        N_Net_pred['shading'] = self.get_shading(N_Net_pred['normal'], L_Net1_pred['dirs'])
        pred['prev_shading'] = torch.cat(N_Net_pred['shading'], 1)

        L_Net2_inputs = self.prepare_LNet2_inputs(L_Net1_inputs, L_Net1_pred, N_Net_pred)
        L_Net2_pred = self.L_Net2(L_Net2_inputs)
        for k in L_Net2_pred: 
            pred[k] = L_Net2_pred[k]

        return pred
