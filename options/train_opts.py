from .base_opts import BaseOpts
class TrainOpts(BaseOpts):
    def __init__(self):
        super(TrainOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        #### Training Arguments ####
        self.parser.add_argument('--dataset', default='ups_synth_dataset')
        self.parser.add_argument('--data_dir', default='data/datasets/PS_Blobby_Dataset')
        self.parser.add_argument('--data_dir2', default='data/datasets/PS_Sculpture_Dataset')
        self.parser.add_argument('--concat_data', default=True, action='store_false')
        self.parser.add_argument('--test_h', default=128, type=int) # For lighting estimation
        self.parser.add_argument('--test_w', default=128, type=int)
        self.parser.add_argument('--milestones', default=[5, 10, 15], nargs='+', type=int)
        self.parser.add_argument('--start_epoch', default=1, type=int)
        self.parser.add_argument('--epochs', default=20, type=int)
        self.parser.add_argument('--batch', default=32, type=int)
        self.parser.add_argument('--val_batch', default=8, type=int)
        self.parser.add_argument('--init_lr', default=0.0005, type=float)
        self.parser.add_argument('--lr_decay', default=0.5, type=float)
        self.parser.add_argument('--beta_1', default=0.9, type=float, help='adam')
        self.parser.add_argument('--beta_2', default=0.999, type=float, help='adam')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='sgd')
        self.parser.add_argument('--w_decay', default=4e-4, type=float)

        #### Loss Arguments ####
        self.parser.add_argument('--normal_w', default=1, type=float)
        self.parser.add_argument('--dir_w', default=1, type=float)
        self.parser.add_argument('--ints_w', default=1, type=float)
        self.parser.add_argument('--sd_w', default=1, type=float)
        self.parser.add_argument('--spec_sd_w', default=1, type=float) # Not used
        self.is_train = True

    def collect_info(self): 
        BaseOpts.collect_info(self)
        self.str_keys += []
        self.val_keys  += ['batch', 'init_lr', 'normal_w', 'dir_w', 'ints_w']
        self.bool_keys += ['concat_data'] 
        self.args.str_keys = self.str_keys
        self.args.val_keys = self.val_keys
        self.args.bool_keys = self.bool_keys

    def set_default(self):
        BaseOpts.set_default(self)
        self.collect_info()

    def parse(self):
        BaseOpts.parse(self)
        self.set_default()
        return self.args
