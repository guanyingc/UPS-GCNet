from .base_opts import BaseOpts


class RunModelOpts(BaseOpts):
    def __init__(self):
        super(RunModelOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        self.parser.add_argument('--run_model', default=True, action='store_false')
        self.parser.add_argument('--benchmark', default='ups_DiLiGenT_dataset')
        self.parser.add_argument('--bm_dir', default='data/datasets/DiLiGenT/pmsData_crop')
        self.parser.add_argument('--test_root', default='')
        self.parser.add_argument('--test_resc', default=True, action='store_false')
        self.parser.add_argument('--test_h', default=128, type=int)
        self.parser.add_argument('--test_w', default=128, type=int)
        self.parser.add_argument('--repeat', default=1, type=int)
        self.parser.add_argument('--epochs', default=1, type=int)
        self.parser.add_argument('--test_batch', default=1, type=int) 
        self.parser.add_argument('--test_disp', default=1, type=int)
        self.parser.add_argument('--test_save', default=1, type=int)
        self.parser.add_argument('--max_test_iter', default=-1, type=int)
        self.parser.add_argument('--test_intv', default=1, type=int) 
        self.parser.add_argument('--save_detail', default=True, action='store_false')
        self.parser.add_argument('--est_num', default=1, type=int)
        self.parser.add_argument('--syn_obj', default='_sphere')
        self.parser.add_argument('--light_index', default=None)
        self.is_train = False

    def collect_info(self):
        self.str_keys += ['model', 'benchmark', 'fuse_type']
        self.val_keys += ['in_img_num', 'test_h', 'test_w']
        self.bool_keys += ['save_detail', 'int_aug', 'test_resc']
        self.args.str_keys = self.str_keys
        self.args.val_keys = self.val_keys
        self.args.bool_keys = self.bool_keys

    def set_default(self):
        self.collect_info()

    def parse(self):
        BaseOpts.parse(self)
        self.set_default()
        return self.args
