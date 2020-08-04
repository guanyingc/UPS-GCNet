import datetime, time, os
import numpy as np
import torch
import torchvision.utils as vutils
import scipy.io as sio
from . import utils
from . import draw_utils # TODO: clean

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')
plt.rcParams["figure.figsize"] = (5,8)

class Logger(object):
    def __init__(self, args):
        self.times = {'init': time.time()}
        if args.make_dir:
            self._setup_dirs(args)
        self.args = args
        self.print_args()

    def print_args(self):
        strs = '------------ Options -------------\n'
        strs += '{}'.format(utils.dict_to_string(vars(self.args)))
        strs += '-------------- End ----------------\n'
        self.print_write(strs)
    
    def print_write(self, strs):
        print('%s' % strs)
        if self.args.make_dir:
            self.log_fie.write('%s\n' % strs)
            self.log_fie.flush()

    # Directory related
    def _add_arguments(self, args):
        info = ''
        arg_var  = vars(args)
        if hasattr(args, 'run_model') and args.run_model:
            test_root = arg_var['test_root']
            info += ',%s' % os.path.basename(arg_var[test_root]).split('.')[0]
        for k in args.str_keys:  
            info = '{0},{1}'.format(info, arg_var[k])
        for k in args.val_keys:  
            var_key = k[:2] + '_' + k[-1]
            info = '{0},{1}-{2}'.format(info, var_key, arg_var[k])
        for k in args.bool_keys: 
            info = '{0},{1}'.format(info, k) if arg_var[k] else info 
        return info

    def _setup_dirs(self, args):
        date_now = datetime.datetime.now()
        self.date = '%d-%d' % (date_now.month, date_now.day)
        dir_name = self.date
        dir_name += (',%s' % args.suffix) if args.suffix else ''
        dir_name += self._add_arguments(args) 
        dir_name += ',DEBUG' if args.debug else ''

        self._check_path(args, dir_name)
        file_dir = os.path.join(args.log_dir, '%s,%s' % (dir_name, date_now.strftime('%H:%M:%S')))
        self.log_fie = open(file_dir, 'w')
        return 

    def _check_path(self, args, dir_name):
        if hasattr(args, 'run_model') and args.run_model:
            test_root = vars(args)[args.test_root]
            log_root = os.path.join(os.path.dirname(test_root), dir_name)
            args.log_dir = log_root
            sub_dirs = ['test']
        else:
            if args.debug:
                dir_name = 'DEBUG/' + dir_name
            log_root = os.path.join(args.save_root, args.dataset, args.item, dir_name)
            args.log_dir = os.path.join(log_root, 'logdir')
            args.cp_dir  = os.path.join(log_root, 'checkpointdir')
            utils.make_files([args.log_dir, args.cp_dir])
            sub_dirs = ['train', 'val']

        for sub_dir in sub_dirs:
            utils.make_files([os.path.join(args.log_dir, sub_dir, 'Images')])
    
    # Print related
    def get_time_info(self, epoch, iters, batch):
        time_elapsed = (time.time() - self.times['init']) / 3600.0
        total_iters  = (self.args.epochs - self.args.start_epoch + 1) * batch
        cur_iters = (epoch - self.args.start_epoch) * batch + iters
        time_total = time_elapsed * (float(total_iters) / cur_iters)
        return time_elapsed, time_total

    def print_iters_summary(self, opt):
        epoch, iters, batch = opt['epoch'], opt['iters'], opt['batch']
        strs = ' | {}'.format(str.upper(opt['split']))
        strs += ' Iter [{}/{}] Epoch [{}/{}]'.format(iters, batch, epoch, self.args.epochs)

        if opt['split'] == 'train': 
            time_elapsed, time_total = self.get_time_info(epoch, iters, batch) # Buggy for test
            strs += ' Clock [{:.2f}h/{:.2f}h]'.format(time_elapsed, time_total)
            strs += ' LR [{}]'.format(opt['recorder'].records[opt['split']]['lr'][epoch][0])
        self.print_write(strs)

        if 'timer' in opt.keys(): 
            self.print_write(opt['timer'].time_to_string())

        if 'recorder' in opt.keys(): 
            self.print_write(opt['recorder'].iter_rec_to_string(opt['split'], epoch))

    def print_epoch_summary(self, opt):
        split = opt['split']
        epoch = opt['epoch']
        self.print_write('---------- {} Epoch {} Summary -----------'.format(str.upper(split), epoch))
        self.print_write(opt['recorder'].epoch_rec_to_string(split, epoch))
    
    # Tensor processing related
    def save_img_results(self, results, split, epoch, iters, nrow, error=''):
        max_save_n = self.args.test_save_n if split == 'test' else self.args.train_save_n
        res = [img.cpu() for img in results]
        res = self.split_multi_channel(res, max_save_n)
        res = torch.cat(self.convert_to_same_size(res), 0)
        save_dir = self.get_save_dir(split, epoch)
        save_prefix = os.path.join(save_dir, '%02d_%03d' % (epoch, iters))
        save_prefix += ('_%s' % error) if error != '' else ''
        if self.args.save_split: 
            self.save_split(res, save_prefix)
        else:
            vutils.save_image(res, save_prefix + '_out.jpg', nrow=nrow, pad_value=1)

    def get_save_dir(self, split, epoch):
        save_dir = os.path.join(self.args.log_dir, split, 'Images')
        run_model = hasattr(self.args, 'run_model') and self.args.run_model
        if not run_model and epoch > 0:
            save_dir = os.path.join(save_dir, str(epoch))
        utils.make_file(save_dir)
        return save_dir

    def split_multi_channel(self, t_list, max_save_n = 8):
        # Split the tensor to multiple 3-channel tensor if the channel number larger than 3
        new_list = []
        for tensor in t_list:
            if tensor.shape[1] > 3:
                num = 3
                new_list += torch.split(tensor, num, 1)[:max_save_n]
            else:
                new_list.append(tensor)
        return new_list

    def convert_to_same_size(self, t_list):
        shape = (t_list[0].shape[0], 3, t_list[0].shape[2], t_list[0].shape[3])
        for i, tensor in enumerate(t_list):
            n, c, h, w = tensor.shape
            if tensor.shape[1] != shape[1]: # check channel
                tensor = tensor.expand((n, shape[1], h, w))
                t_list[i] = tensor
            if h == shape[2] and w == shape[3]:
                continue
            t_list[i] = torch.nn.functional.interpolate(tensor, [shape[2], shape[3]], mode='bilinear', align_corners=False)
        return t_list

    def save_split(self, res, save_prefix):
        n, c, h, w = res.shape
        for i in range(n):
            vutils.save_image(res[i], save_prefix + '_%d.png' % (i))
    
    # Plot related 
    def plot_curves(self, recorder, split='train', epoch=-1, intv=1):
        dict_of_array = recorder.record_to_dict_of_array(split, epoch, intv)
        save_dir = os.path.join(self.args.log_dir, split)
        if epoch < 0:
            save_dir = self.args.log_dir
            save_name = '%s_Summary.jpg' % (split)
        else:
            save_name = '%s_epoch_%d.jpg' % (split, epoch)

        classes = ['loss', 'acc', 'err', 'lr', 'ratio']
        classes = utils.check_in_list(classes, dict_of_array.keys())
        if len(classes) == 0: return

        for idx, c in enumerate(classes):
            plt.subplot(len(classes), 1, idx+1)
            plt.grid()
            legends = []
            for k in dict_of_array.keys():
                if (c in k.lower()) and not k.endswith('_x'):
                    plt.plot(dict_of_array[k+'_x'], dict_of_array[k])
                    legends.append(k)
            if len(legends) != 0:
                plt.legend(legends, bbox_to_anchor=(0.5, 1.05), loc='upper center', 
                            ncol=3, prop=fontP)
                plt.title(c)
                if epoch < 0: plt.xlabel('Epoch') 
                else: plt.xlabel('Iters')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.clf()

    def config_save_detail_dir(self, split, epoch):
        save_detail_dir = os.path.join(self.args.log_dir, split, 'Details', str(epoch))
        utils.make_file(save_detail_dir)
        return save_detail_dir

    def plot_lighting(self, dirs, ints, obj_name, save_dir):
        # Visualize light direction and intensity
        save_name = os.path.join(save_dir, obj_name + '_lighting.jpg') 
        ints = ints / ints.max()
        draw_utils.plot_light(dirs[:,0], dirs[:, 1], save_name, ints)

    def plot_dir_error(self, light, error, obj_name, save_dir):
        # plot light direction estimation error
        save_name = os.path.join(save_dir, obj_name + '_error_dir.jpg') 
        error = error / 30
        draw_utils.plot_light(light[:,0], light[:, 1], save_name, error)

    def plot_int_error(self, light, error, obj_name, save_dir):
        # plot light intensity estimation error
        save_name = os.path.join(save_dir, obj_name + '_error_int.jpg') 
        error = error / 0.5
        draw_utils.plot_light(light[:,0], light[:, 1], save_name, error)

    def save_normal_mat(self, normal_est, obj_name, save_dir):
        # Save the estimated normal to matlab matrix file
        normals = normal_est.cpu()
        batch = normals.shape[0]
        for i in range(batch):
            save_name = '%s_Normal_%s' % (obj_name[i], 'ECCV2020')
            mask = (np.square(normals[i]).sum(0) > 0.01).expand_as(normals[i])
            n_img = (normals[i]+1)/2 * mask.float() + (1- mask.float())
            vutils.save_image(n_img, os.path.join(save_dir, save_name + '.png'))
            normal = normals[i].numpy().transpose(1, 2, 0)
            sio.savemat(os.path.join(save_dir, save_name + '.mat'), {'Normal_est': normal}, do_compression=True)

    def save_light(self, dirs, save_path):
        # Save the estimated light direction to txt file
        dirs = dirs.squeeze()
        np.savetxt(os.path.join(save_path[0], 'GCNet_light_dirs_%s.txt' % self.date), dirs, fmt='%.6f')

    def save_intens(self, ints, save_path):
        # Save the estimated light intensity to txt file
        if ints.dim() == 1:
            ints = ints.view(1, -1)
        img_num = ints.shape[1] // 3 
        ints = np.concatenate(np.split(ints, img_num, 1))
        np.savetxt(os.path.join(save_path[0], 'GCNet_light_ints_%s.txt' % self.date), ints, fmt='%.6f')
        return ints
