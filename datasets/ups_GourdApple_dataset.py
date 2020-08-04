from __future__ import division
import os
import numpy as np
from imageio import imread
import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class UpsGourdAppleDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = os.path.join(args.bm_dir)
        self.split  = split
        self.args   = args
        self.objs   = util.read_list(os.path.join(self.root, 'objects.txt'), sort=False)
        print('[%s Data] \t%d objs.  Root: %s' % (split, len(self.objs), self.root))

        self.imgs, self.normals, self.masks  = {}, {}, {}
        self.lights, self.intens = {}, {}
        for i in range(len(self.objs)):
            if self.args.debug and i >= self.args.max_test_iter:
                break
            self._load_data(i)

    def _get_mask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def _load_data(self, index):
        obj = self.objs[index]
        lights = np.genfromtxt(os.path.join(self.root, obj, 'light_directions_refined.txt'))
        intens = np.genfromtxt(os.path.join(self.root, obj, 'light_intensities_refined.txt'))
        self.lights[obj] = lights
        self.intens[obj] = intens 

        cache_name = os.path.join(self.root, obj, obj + '_uncalib.npy')
        img = np.load(cache_name)
        print('Loaded cache from %s, shape: [%d, %d, %d]' % (cache_name, img.shape[0], img.shape[1], img.shape[2]))
        # The shape of img is [H, W, 3 * N], N is the number of lightings

        img = img / img.max() 
        mask = self._get_mask(obj)
        print('Loaded cache from %s' % (cache_name))

        if self.args.test_resc:
            h, w = self.args.test_h, self.args.test_w
            img  = pms_transforms.rescale_single(img, [h, w])
            mask = pms_transforms.rescale_single(mask, [h, w], 1)

        h, w, c = img.shape
        normal = np.zeros((h, w, 3)) # dummy normal
        normal[:, :, 2] = 1

        k = 4
        img = pms_transforms.imgsize_to_factor_of_k(img, k)
        mask = pms_transforms.imgsize_to_factor_of_k(mask, k)
        normal = pms_transforms.imgsize_to_factor_of_k(normal, k)

        img = img * mask.repeat(img.shape[2], 2)
        self.imgs[obj], self.normals[obj], self.masks[obj] = img, normal, mask

    def __getitem__(self, index):
        obj   = self.objs[index // self.args.repeat]
        item = {'normal': self.normals[obj], 'img': self.imgs[obj], 'mask':self.masks[obj]}

        h, w, c = self.imgs[obj].shape
        proxys = pms_transforms.get_zero_proxy(self.args, h, w, c // 3)
        for k in proxys: item[k] = proxys[k]

        for k in item.keys(): 
            item[k] = pms_transforms.array_to_tensor(item[k])

        item['dirs'] = torch.from_numpy(self.lights[obj]).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(self.intens[obj]).view(-1, 1, 1).float()
        item['obj'] = obj
        item['path'] = os.path.join(self.root, obj)
        return item

    def __len__(self):
        return len(self.objs) * self.args.repeat
