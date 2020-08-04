from __future__ import division
import os
import numpy as np
import OpenEXR

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class UpsSynthTestDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.root = os.path.join(args.bm_dir)
        self.split = split
        self.args = args
        self.shape_list = util.read_list(os.path.join(self.root, '%s_mtrl_%s.txt' % 
            (split, self.args.syn_obj)), sort=False)
        self.repeat = 1

    def _get_input_path(self, index):
        index = index // self.repeat
        shape, mtrl = self.shape_list[index].split('/')
        normal_path = os.path.join(self.root, 'EXR', shape + '.exr')
        if not os.path.exists(normal_path):
            normal_path = os.path.join(self.root, 'EXR', shape, shape + '.exr')
        img_dir = os.path.join(self.root, 'Images', self.shape_list[index])
        exr_dir = os.path.join(self.root, 'EXR', self.shape_list[index])
        img_list = util.read_list(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))

        data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        dirs = data[:, 1:4].astype(np.float32)

        if hasattr(self.args, 'light_index') and self.args.light_index != None:
            index_path = os.path.join(self.root, 'Lighting_Index', self.args.light_index)
            select_idx = np.genfromtxt(index_path, dtype=int)
        else:
            select_idx = np.array(range(data.shape[0]))[:100]
        print('Image number: %d' % (len(select_idx)))
        data = data[select_idx, :]
        imgs = [os.path.join(exr_dir, img[:-8] + '.exr') for img in data[:, 0]]
        dirs = data[:, 1:4].astype(np.float32)
        return normal_path, imgs, dirs

    def __getitem__(self, index):
        np.random.seed(index)
        normal_path, img_list, dirs = self._get_input_path(index)
        normal = util.exr_to_array(OpenEXR.InputFile(normal_path), 'normal')

        imgs = []
        for i in img_list:
            img = util.exr_to_array(OpenEXR.InputFile(i), 'color')
            imgs.append(img)

        img = np.concatenate(imgs, 2)
        h, w, c = img.shape

        mask = pms_transforms.normal_to_mask(normal, 0.2)
        normal = normal * mask.repeat(3, 2)

        # As the image intensities of the dark materials are very small, we Scale up the magnitude of the synthetic images
        ratio = mask.sum().astype(np.float) / (mask.shape[0] * mask.shape[1]) # ratio of object area in the whole image
        thres = 0.02
        if img.mean() / ratio < thres: # if the mean value of the object region less than 0.02
            # scale the mean value of the object region to thres (i.e., 0.02)
            img *= thres / (img.mean() / ratio)
        img = (img * 1.5).clip(0, 2)

        if self.args.int_aug: # and not no_int_aug:
            ints = pms_transforms.get_intensity(len(imgs))
            img = np.dot(img, np.diag(ints.reshape(-1)))
        else:
            ints = np.ones(c)

        if self.args.test_resc:
            img, normal = pms_transforms.rescale(img, normal, [self.args.test_h, self.args.test_w])
            mask = pms_transforms.rescale_single(mask, [self.args.test_h, self.args.test_w])

        norm = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10)

        item = {'normal': normal, 'img': img, 'mask': mask}
        proxys = pms_transforms.get_proxy_features(self.args, normal, dirs)
        for k in proxys: 
            item[k] = proxys[k]

        for k in item.keys(): 
            item[k] = pms_transforms.array_to_tensor(item[k])

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(ints).view(-1, 1, 1).float()
        item['obj'] = '_'.join(self.shape_list[index // self.repeat].split('/'))
        item['path'] = os.path.join(self.root, 'Images', self.shape_list[index // self.repeat])
             
        return item

    def __len__(self):
        return len(self.shape_list) * self.repeat

