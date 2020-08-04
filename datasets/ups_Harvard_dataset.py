import os
import numpy as np
from imageio import imread
import scipy.io as sio

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class UpsHarvardDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = os.path.join(args.bm_dir)
        self.split  = split
        self.args   = args
        self.objs   = util.read_list(os.path.join(self.root, 'objects.txt'), sort=False)
        self.names  = util.read_list(os.path.join(self.root, 'names.txt'))
        print('[%s Data] \t%d objs %d lights. Root: %s' % (split, len(self.objs), len(self.names), self.root))

    def __getitem__(self, index):
        obj   = self.objs[index]
        select_idx = np.random.permutation(len(self.names))[:self.args.in_img_num]
        img_list   = [os.path.join(self.root, obj, 'Objects', self.names[i]) for i in select_idx]

        if obj in ['cat']:
            dirs = np.genfromtxt(os.path.join(self.root, obj, 'refined_light.txt'), dtype='float', delimiter=',')
        else:
            dirs = np.genfromtxt(os.path.join(self.root, obj, 'refined_light.txt'), dtype='float')
        dirs = dirs.transpose()[select_idx]

        normal_path = os.path.join(self.root, obj, 'result.mat')
        normal = sio.loadmat(normal_path)['n']
        normal[np.isnan(normal)] = 0
        normal = np.array(np.flip(normal, 0))
        
        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            hw = img.shape
            t_img = img.reshape(hw[0], hw[1], 1)
            img = np.repeat(t_img, 3, 2)
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        mask = pms_transforms.normal_to_mask(normal)

        if self.args.test_resc:
            img, normal = pms_transforms.rescale(img, normal, [self.args.test_h, self.args.test_w])
            mask = pms_transforms.rescale_single(mask, [self.args.test_h, self.args.test_w], 0)
        img  = img * mask.repeat(img.shape[2], 2)

        item = {'normal': normal, 'img': img, 'mask': mask}

        downsample = 4
        for k in item.keys():
            item[k] = pms_transforms.imgsize_to_factor_of_k(item[k], downsample)

        for k in item.keys(): 
            item[k] = pms_transforms.array_to_tensor(item[k])

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['ints'] = torch.ones(dirs.shape).view(-1, 1, 1).float()
        item['obj'] = obj
        item['path'] = os.path.join(self.root, obj)
        return item

    def __len__(self):
        return len(self.objs) 
