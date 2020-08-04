import torch
from .base_model import BaseModel
from .L_model import LModel
from utils import eval_utils
from . import model_utils

class GCNetModel(LModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--L_Net1_name',    default='L_Net') # specify the name of the subnets and checkpoint
        parser.add_argument('--L_Net1_checkp',  default='')
        parser.add_argument('--N_Net_name',     default='N_Net')
        parser.add_argument('--N_Net_checkp',   default='')
        parser.add_argument('--L_Net2_name',    default='L_Net')
        parser.add_argument('--L_Net2_checkp',  default='')
        parser.add_argument('--GCNet_name',   default='GCNet')
        parser.add_argument('--GCNet_checkp', default='') # Specify the checkpoint of the trained GCNet

        parser.add_argument('--in_est_n',       default=True,  action='store_false') # input for L-Net2
        parser.add_argument('--in_est_sd',      default=True,  action='store_false')
        parser.add_argument('--end2end_ft',     default=False, action='store_true')

        if is_train:
            parser.set_defaults(milestones=[5, 10, 15], epochs=20, init_lr=0.001) 
        else:
            parser.set_defaults(test_resc=False, test_root='GCNet_checkp') 
            # In run_model mode, results will be save in the same directory as args.GCNet_checkp

        str_keys = ['L_Net1_name', 'N_Net_name', 'L_Net2_name', 'GCNet_name']
        val_keys = [ ]
        bool_keys = ['in_est_n', 'in_est_sd', 'end2end_ft'] 
        return parser, str_keys, val_keys, bool_keys

    def __init__(self, args, log):
        opt = vars(args)
        BaseModel.__init__(self, opt)
        self.net_names = ['GCNet']

        c_in = self.get_in_channel_nums(opt, log)
        self.GCNet = self.import_network(args.GCNet_name)(opt, c_in)
        self.GCNet = model_utils.init_net(self.GCNet, gpu_ids=args.gpu_ids)

        network = self.GCNet.module if isinstance(self.GCNet, torch.nn.DataParallel) else self.GCNet

        for name in ['L_Net1', 'N_Net', 'L_Net2']: # Load checkpoint for the sub-networks
            checkp_path = self.opt['%s_checkp' % name]
            if checkp_path != '':
                log.print_write("==> [%s] loading pre-trained model from %s" % (name, checkp_path))
                subnet = getattr(network, name)
                self._load_checkpoint(subnet, checkp_path, log)
        
        if not args.end2end_ft:
            L_Net1, N_Net = network.L_Net1, network.N_Net
            self.set_requires_grad(L_Net1, requires_grad=False)
            self.set_requires_grad(N_Net, requires_grad=False)

        if self.is_train: # Criterion
            self.dir_crit = model_utils.DirectionCrit(opt, log) 
            self.int_crit = model_utils.IntensityCrit(opt, log)
            parameters = filter(lambda p: p.requires_grad, self.GCNet.parameters())
            self.optimizer = torch.optim.Adam(parameters, lr=args.init_lr, betas=(args.beta_1, args.beta_2))
            self.optimizers.append(self.optimizer)
            self.setup_lr_scheduler()

            if args.end2end_ft:
                self.normal_crit = model_utils.NormalCrit(opt, log)
                if self.opt['in_est_sd']:
                    self.shading_crit = torch.nn.MSELoss() 
        self.load_checkpoint(log)

    def get_in_channel_nums(self, opt, log):
        c_in = 0
        name_nums = {'in_img': 3, 'in_mask': 1}
        for k in name_nums.keys():
            if opt[k]:
                log.print_write('==> [Network Input]: Adding %s as input [%d]' % (k, name_nums[k]))
                c_in += name_nums[k]
        return c_in

    def forward(self):
        self.pred = self.GCNet(self.data)
        return self.pred

    def optimize_weights(self):
        dir_loss, dir_loss_term = self.dir_crit(self.pred['dirs_x'], self.pred['dirs_y'], self.data['dirs'])
        ints_loss, ints_loss_term = self.int_crit(self.pred['ints'], self.data['ints'])
        self.loss = self.opt['dir_w'] * dir_loss + self.opt['ints_w'] * ints_loss

        self.loss_terms = {}
        for k in dir_loss_term: 
            self.loss_terms[k] = dir_loss_term[k]
        for k in ints_loss_term: 
            self.loss_terms[k] = ints_loss_term[k]

        if self.opt['end2end_ft']:
            dir_loss1, dir1_loss_term = self.dir_crit(self.pred['prev_dirs_x'], self.pred['prev_dirs_y'], self.data['dirs'])
            ints_loss1, ints1_loss_term = self.int_crit(self.pred['prev_ints'], self.data['ints'])
            self.loss += self.opt['dir_w'] * dir_loss1 + self.opt['ints_w'] * ints_loss1 

            normal_loss, normal_loss_term = self.normal_crit(self.pred['prev_normal'], self.data['normal'])
            self.loss += self.opt['normal_w'] * normal_loss

            for k in dir1_loss_term: 
                self.loss_terms['p_%s' % k] = dir1_loss_term[k]
            for k in ints1_loss_term: 
                self.loss_terms['p_%s' % k] = ints1_loss_term[k]
            for k in normal_loss_term:
                self.loss_terms[k] = normal_loss_term[k]

            if self.opt['in_est_sd']: 
                sd_loss = self.shading_crit(self.pred['prev_shading'], self.data['shading'])
                self.loss += self.opt['sd_w'] * sd_loss
                self.loss_terms['p_shading_loss'] = sd_loss.item()
            
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def prepare_records(self):
        records, iter_res = LModel.prepare_records(self)

        data, pred = self.data, self.pred
        #if 'prev_dirs' in pred:
        #    prev_l_acc, _ = eval_utils.cal_dirs_acc(data['dirs'].detach(), pred['prev_dirs'].detach())
        #    records['p_l_err_mean'] = prev_l_acc['l_err_mean']
        #    iter_res.append(prev_l_acc['l_err_mean'])

        #if 'prev_intens' in pred:
        #    prev_int_acc, _ = eval_utils.cal_ints_acc(data['ints'].detach(), pred['prev_intens'].detach())
        #    records['p_ints_ratio'] = prev_int_acc['ints_ratio']
        #    iter_res.append(prev_int_acc['ints_ratio'])

        if ('prev_normal' in pred) and (data['normal'].shape == pred['prev_normal'].shape):
            n_acc, _ = eval_utils.cal_normal_acc(data['normal'].detach(), pred['prev_normal'].detach(), data['mask'].detach())
            records['p_n_err_mean'] = n_acc['n_err_mean']
            iter_res.append(n_acc['n_err_mean'])

        if ('shading' in pred) and ('shading' in data) and (data['shading'].shape == pred['shading'].shape):
            records['sd_loss'] = eval_utils.cal_mse(data['shading'].detach(), pred['shading'].detach()) 

        if ('prev_shading' in pred) and ('shading' in data) and (data['shading'].shape == pred['prev_shading'].shape):
            records['p_sd_loss'] = eval_utils.cal_mse(data['shading'].detach(), pred['prev_shading'].detach()) 
        return records, iter_res

    def prepare_visual(self):
        visuals = []
        visuals += LModel.prepare_visual(self)
        data, pred = self.data, self.pred
        if 'prev_normal' in pred:
            visuals.append((pred['prev_normal'].detach() + 1) / 2.0)
        if self.opt['in_est_sd']:
            visuals.append(pred['prev_shading'].narrow(1, 0, 6))
        return visuals
