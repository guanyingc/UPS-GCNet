import torch
import random
import numpy as np
import cv2
random.seed(0)
np.random.seed(0)

def array_to_tensor(array):
    if array is None:
        return array
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return tensor.float()

def normal_to_mask(normal, thres=1e-2):
    mask = (np.square(normal).sum(2, keepdims=True) > thres).astype(np.float32)
    return mask

def imgsize_to_factor_of_k(img, k):
    if img.shape[0] % k == 0 and img.shape[1] % k == 0:
        return img
    pad_h, pad_w = k - img.shape[0] % k, k - img.shape[1] % k
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), 
            'constant', constant_values=((0,0),(0,0),(0,0)))
    return img

def random_crop(inputs, target, size):
    h, w, _ = inputs.shape
    c_h, c_w = size
    if h == c_h and w == c_w:
        return inputs, target
    x1 = random.randint(0, w - c_w)
    y1 = random.randint(0, h - c_h)
    inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
    target = target[y1: y1 + c_h, x1: x1 + c_w]
    return inputs, target

def rescale(inputs, target, size):
    in_h, in_w, _ = inputs.shape
    h, w = size
    if h != in_h or w != in_w:
        inputs = cv2.resize(inputs, tuple(size), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, tuple(size), interpolation=cv2.INTER_LINEAR)
    return inputs, target

def rescale_single(inputs, size, order=1):
    in_h, in_w, _ = inputs.shape
    h, w = size
    if h != in_h or w != in_w:
        inputs = cv2.resize(inputs, tuple(size), interpolation=cv2.INTER_LINEAR)
    return inputs.reshape(h, w, -1)

def random_noise_aug(inputs, noise_level=0.05):
    noise = np.random.random(inputs.shape)
    noise = (noise - 0.5) * noise_level
    inputs += noise
    return inputs

def get_intensity(num):
    intensity = np.random.random((num, 1)) * 1.8 + 0.2
    color = np.ones((1, 3)) # Uniform color
    intens = (intensity.repeat(3, 1) * color)
    return intens

def normalize_to_unit_len(matrix, dim=2):
    denorm = np.sqrt((matrix * matrix).sum(dim, keepdims=True))
    matrix = matrix / (denorm + 1e-10)
    return matrix

def get_zero_proxy(args, h, w, c):
    item = {}
    if args.est_shadow:
        item['shadow'] = np.zeros((h, w, c))
    if args.est_spec_sd:
        item['spec_sd'] = np.zeros((h, w, c))
    if args.est_sd:
        item['shading'] = np.zeros((h, w, c))
    return item

def get_proxy_features(args, normal, dirs):
    item = {}
    n_new = normal
    if args.est_shadow: # attached shadow
        item['shadow'] = _get_shadow(n_new, dirs).astype(np.float32)
    if args.est_spec_sd: # specular shading
        item['spec_sd'] = _get_spec_shading(n_new, dirs).astype(np.float32)
    if args.est_sd: # shading
        item['shading'] = _get_shading(n_new, dirs).astype(np.float32)
    return item

def _get_shadow(normal, dirs):
    h, w, c = normal.shape
    shadow = np.dot(normal.reshape(h * w, 3), dirs.transpose()) <= 0
    mask = normal_to_mask(normal)
    shadow = shadow.reshape(h, w, -1) * mask
    return shadow

def _get_shading(normal, dirs):
    h, w, c = normal.shape
    shading = np.dot(normal.reshape(-1, 3), dirs.transpose()).clip(0)
    shading = shading.reshape(h, w, -1)
    return shading

def _get_spec_shading(normal, dirs):
    h, w, c = normal.shape
    view = np.zeros(dirs.shape).astype(np.float32)
    view[:,2] = 1.0
    bisector = normalize_to_unit_len(view + dirs, dim=1)
    spec_shading = np.dot(normal.reshape(-1, 3), bisector.transpose()).clip(0).reshape(h, w, -1)
    return spec_shading

