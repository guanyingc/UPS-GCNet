import OpenEXR
import os, argparse, sys
import numpy as np
root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(root_path)
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/path/to/dataset/')
parser.add_argument('--obj_list',  default='objects.txt')
parser.add_argument('--mute',       default=True, action='store_false')
parser.add_argument('--normalize',  default=False, action='store_true') # normalize images with light intensity
args = parser.parse_args()

def exr_to_array(I, cname, mute=True):
    hw = I.header()['dataWindow']
    w = hw.max.x+1
    h = hw.max.y+1
    channels = I.header()['channels'].keys()
    prefix = cname + '.' if cname + '.R' in I.header()['channels'].keys() else ''

    r = I.channel(prefix + 'R')
    g = I.channel(prefix + 'G')
    b = I.channel(prefix + 'B')
    R = np.fromstring(r, np.float16).reshape(h, w)
    G = np.fromstring(g, np.float16).reshape(h, w)
    B = np.fromstring(b, np.float16).reshape(h, w)

    img = np.stack([R, G, B], 0).astype(np.float)
    if not mute:
        print('[Normalizing %s], max value: %f, min: %f, mean: %f' % 
                (cname, img.max(), img.min(), img.max()))
    img = img.transpose(1, 2, 0)
    return img

def convert_to_numpy(img_list, intens=None):
    imgs = []
    for idx, img_name in enumerate(img_list):
        if idx % 10 == 0: 
            print('Current idx %d' % (idx))
        img = exr_to_array(OpenEXR.InputFile(img_name), 'color')
        if intens is not None:
            img = np.dot(img, intens[idx])
            save_suffix = '_calib.npy'
        else:
            save_suffix = '_uncalib.npy'
        imgs.append(img)
    img_array = np.concatenate(imgs, 2)
    save_name = os.path.dirname(img_name) + save_suffix
    print('Saving numpy array to %s' % (save_name))
    np.save(save_name, img_array)
    return img_array

def main(args):
    print('Input dir: %s\n' % args.input_dir)
    dir_list  = utils.read_list(os.path.join(args.input_dir, args.obj_list))
    for d in dir_list:
        print("Converting Object: %s" % (d))
        name_list = utils.read_list(os.path.join(args.input_dir, d, 'names.txt'))
        img_list = [os.path.join(args.input_dir, d, name_list[i]) for i in range(len(name_list))]
        intens = None
        if args.normalize:
            intens = np.genfromtxt(os.path.join(args.input_dir, d, 'light_intensities_refined.txt'))
            intens = [np.diag(1 /intens[i]) for i in range(len(name_list))]
        convert_to_numpy(img_list, intens)

if __name__ == '__main__':
    main(args)

