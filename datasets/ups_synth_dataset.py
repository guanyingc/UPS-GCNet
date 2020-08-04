from __future__ import division
import os
import numpy as np
from imageio import imread

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class UpsSynthDataset(data.Dataset):
    def __init__(self, args, root, split='train'):
        self.root = os.path.join(root)
        self.split = split
        self.args = args
        self.shape_list = util.read_list(os.path.join(self.root, split + args.l_suffix))

    def _get_input_path(self, index):
        shape, mtrl = self.shape_list[index].split('/')
        normal_path = os.path.join(self.root, 'Images', shape, shape + '_normal.png')
        img_dir = os.path.join(self.root, 'Images', self.shape_list[index])
        img_list = util.read_list(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))

        data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        select_idx = np.random.permutation(data.shape[0])[:self.args.in_img_num]
        data = data[select_idx, :]
        imgs = [os.path.join(img_dir, img) for img in data[:, 0]]
        dirs = data[:, 1:4].astype(np.float32)
        return normal_path, imgs, dirs

    def __getitem__(self, index):
        normal_path, img_list, dirs = self._get_input_path(index)
        normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
        mask = pms_transforms.normal_to_mask(normal)
        normal = normal * mask.repeat(3, 2) # set background to [0, 0, 0]

        imgs = []
        for i in img_list:
            img = imread(i).astype(np.float32) / 255.0
            imgs.append(img)

        img = np.concatenate(imgs, 2)

        h, w, c = img.shape
        crop_h, crop_w = self.args.crop_h, self.args.crop_w

        if self.args.rescale and not (crop_h == h):
            sc_h = np.random.randint(crop_h, h) if self.args.rand_sc else self.args.scale_h
            sc_w = np.random.randint(crop_w, w) if self.args.rand_sc else self.args.scale_w
            img, normal = pms_transforms.rescale(img, normal, [sc_h, sc_w])

        if self.args.crop:
            img, normal = pms_transforms.random_crop(img, normal, [crop_h, crop_w])

        if self.args.color_aug:
            img = img * np.random.uniform(1, self.args.color_ratio)

        if self.args.int_aug:
            ints = pms_transforms.get_intensity(len(imgs))
            img  = np.dot(img, np.diag(ints.reshape(-1)))
        else:
            ints = np.ones(c)

        if self.args.noise_aug:
            img = pms_transforms.random_noise_aug(img, self.args.noise)

        mask = pms_transforms.normal_to_mask(normal)
        normal = pms_transforms.normalize_to_unit_len(normal, dim=2)
        normal = normal * mask.repeat(3, 2) # set background to [0, 0, 0]

        item = {'normal': normal, 'img': img, 'mask': mask}
        proxys = pms_transforms.get_proxy_features(self.args, normal, dirs)

        for k in proxys: 
            item[k] = proxys[k]

        for k in item.keys(): 
            item[k] = pms_transforms.array_to_tensor(item[k])

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(ints).view(-1, 1, 1).float()
        return item

    def __len__(self):
        return len(self.shape_list)
