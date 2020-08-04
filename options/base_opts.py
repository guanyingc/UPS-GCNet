import argparse
import models
import torch


class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # lists to store important parameters that will be displayed in the logfile name
        self.str_keys = []
        self.val_keys = []
        self.bool_keys = []

    def initialize(self):
        # Training Data Preprocessing Arguments ####
        self.parser.add_argument('--l_suffix', default='_mtrl.txt') # train image list suffix
        self.parser.add_argument('--rescale', default=True, action='store_false')
        self.parser.add_argument('--rand_sc', default=True, action='store_false')
        self.parser.add_argument('--scale_h', default=128, type=int)
        self.parser.add_argument('--scale_w', default=128, type=int)
        self.parser.add_argument('--crop', default=True, action='store_false')
        self.parser.add_argument('--crop_h', default=128, type=int)
        self.parser.add_argument('--crop_w', default=128, type=int)
        self.parser.add_argument('--int_aug', default=True, action='store_false')
        self.parser.add_argument('--noise_aug', default=True, action='store_false')
        self.parser.add_argument('--noise', default=0.05, type=float)
        self.parser.add_argument('--color_aug', default=True, action='store_false')
        self.parser.add_argument('--color_ratio', default=3, type=float)

        # Device Arguments ####
        self.parser.add_argument('--gpu_ids', default='0', help='0,1,2,3 for gpu / -1 for cpu')
        self.parser.add_argument('--time_sync', default=False, action='store_true')
        self.parser.add_argument('--workers', default=8, type=int)
        self.parser.add_argument('--seed', default=0, type=int)

        # Network Arguments ####
        self.parser.add_argument('--dirs_cls', default=36, type=int)
        self.parser.add_argument('--ints_cls', default=20, type=int)
        self.parser.add_argument('--model', default='L_model')
        self.parser.add_argument('--fuse_type', default='max', help='max_mean_var')
        self.parser.add_argument('--in_img_num', default=32, type=int)
        self.parser.add_argument('--in_img', default=True, action='store_false')
        self.parser.add_argument('--in_mask', default=True, action='store_false')
        self.parser.add_argument('--est_sd', default=False, action='store_true')  # shading
        self.parser.add_argument('--est_shadow', default=False, action='store_true')
        self.parser.add_argument('--est_spec_sd', default=False, action='store_true')
        self.parser.add_argument('--use_BN', default=False, action='store_true')
        self.parser.add_argument('--strict', default=True, action='store_false')  # strict matching for checkpoint

        # Displaying Arguments ####
        self.parser.add_argument('--train_disp', default=20, type=int)
        self.parser.add_argument('--train_save', default=200, type=int)
        self.parser.add_argument('--val_intv', default=1, type=int)
        self.parser.add_argument('--val_disp', default=1, type=int)
        self.parser.add_argument('--val_save', default=1, type=int)
        self.parser.add_argument('--max_train_iter',default=-1, type=int)
        self.parser.add_argument('--max_val_iter', default=-1, type=int)
        self.parser.add_argument('--train_save_n', default=4, type=int)
        self.parser.add_argument('--test_save_n', default=4, type=int)
        self.parser.add_argument('--save_split', default=False, action='store_true')
        self.parser.add_argument('--save_intv', default=1, type=int)
        self.parser.add_argument('--save_light', default=False, action='store_true')

        # Log Arguments ####
        self.parser.add_argument('--save_root', default='logdir/')
        self.parser.add_argument('--item', default='GCNet')
        self.parser.add_argument('--suffix', default=None)
        self.parser.add_argument('--debug', default=False, action='store_true')
        self.parser.add_argument('--make_dir', default=True, action='store_false')

    def set_default(self):
        if self.args.debug:
            self.args.train_disp = 1
            self.args.train_save = 1
            self.args.max_train_iter = 4
            self.args.max_val_iter = 4
            self.args.max_test_iter = 4
            self.args.test_intv = 1

    def collect_info(self):
        self.str_keys += ['model', 'fuse_type']
        self.val_keys += ['scale_h', 'crop_h', 'in_img_num', 'dirs_cls', 'ints_cls']
        self.bool_keys += ['use_BN', 'in_mask', 'est_sd', 'est_shadow', 'est_spec_sd', 'color_aug', 'int_aug']

    def gather_options(self):
        args, _ = self.parser.parse_known_args()

        # Update model-related parser options
        model_option_setter = models.get_option_setter(args.model)
        parser, str_keys, val_keys, bool_keys = model_option_setter(self.parser, self.is_train)
        args, _ = parser.parse_known_args()
        self.parser = parser

        self.str_keys += str_keys
        self.val_keys += val_keys
        self.bool_keys += bool_keys
        return parser.parse_args()

    def parse(self):
        args = self.gather_options()
        args.is_train = self.is_train

        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])
        self.args = args
        return self.args
