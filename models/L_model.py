import torch
from .base_model import BaseModel
from utils import eval_utils
from . import model_utils
from collections import OrderedDict

class LModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--L_Net_name',  default='L_Net')
        parser.add_argument('--L_Net_checkp', default='')
        str_keys = ['L_Net_name']
        val_keys = [ ]
        bool_keys = [] 
        if not is_train:
            parser.set_defaults(test_root='L_Net_checkp') 
            # In run_model mode, results will be save in the same directory as args.LNet_name
        return parser, str_keys, val_keys, bool_keys

    def __init__(self, args, log):
        opt = vars(args)
        BaseModel.__init__(self, opt)
        self.net_names = ['L_Net']

        c_in = self.get_in_channel_nums(opt, log)
        self.L_Net = self.import_network(args.L_Net_name)(opt, c_in)
        self.L_Net = model_utils.init_net(self.L_Net, gpu_ids=args.gpu_ids)

        if self.is_train:
            self.dir_crit = model_utils.DirectionCrit(opt, log) 
            self.int_crit = model_utils.IntensityCrit(opt, log)
            self.optimizer = torch.optim.Adam(self.L_Net.parameters(), lr=args.init_lr, betas=(args.beta_1, args.beta_2))
            self.optimizers.append(self.optimizer)
            self.setup_lr_scheduler()

        self.load_checkpoint(log)

    def get_in_channel_nums(self, opt, log):
        c_in = 0
        name_nums = {'in_img': 3, 'in_mask': 1}
        for k in name_nums.keys():
            if opt[k]:
                log.print_write('==> [Network Input]: Adding %s as input [%d]' % (k, name_nums[k]))
                c_in += name_nums[k]
        return c_in

    def prepare_inputs(self, data):
        n, c, h, w = data['img'].shape
        t_h, t_w = self.opt['test_h'], self.opt['test_w']
        if (h == t_h and w == t_w):
            imgs = data['img'] 
        else:
            # print('Rescaling images: from %dX%d to %dX%d' % (h, w, t_h, t_w))
            imgs = torch.nn.functional.interpolate(data['img'], size=(t_h, t_w), mode='bilinear', align_corners=False)

        inputs = list(torch.split(imgs, 3, 1))
        keys = OrderedDict({'in_mask': 'mask'})
        for k in keys:
            if not self.opt[k]: 
                continue
            input_map = data[keys[k]]
            if (input_map.shape[2] != t_h or input_map.shape[3] != t_w):
                input_map = torch.nn.functional.interpolate(input_map, size=(t_h, t_w), mode='bilinear', align_corners=False)
            if k in ['in_mask']:
                for i in range(len(inputs)):
                    inputs[i] = torch.cat([inputs[i], input_map], 1)
        return inputs

    def forward(self):
        self.inputs = self.prepare_inputs(self.data) 
        self.pred = self.L_Net(self.inputs)
        return self.pred

    def optimize_weights(self):
        dir_loss, dir_loss_term = self.dir_crit(self.pred['dirs_x'], self.pred['dirs_y'], self.data['dirs'])
        ints_loss, ints_loss_term = self.int_crit(self.pred['ints'], self.data['ints'])
        self.loss = self.opt['dir_w'] * dir_loss + self.opt['ints_w'] * ints_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.loss_terms = {}
        for k in dir_loss_term: self.loss_terms[k] = dir_loss_term[k]
        for k in ints_loss_term: self.loss_terms[k] = ints_loss_term[k]
    
    def prepare_records(self):
        data, pred = self.data, self.pred
        records = OrderedDict()
        iter_res = []
        if 'dirs' in pred:
            l_acc, pred['dir_err'] = eval_utils.cal_dirs_acc(data['dirs'].detach(), pred['dirs'].detach())
            for k in l_acc: records[k] = l_acc[k]
            iter_res.append(l_acc['l_err_mean'])

        if 'ints' in pred:
            int_acc, pred['int_err'] = eval_utils.cal_ints_acc(data['ints'].detach(), pred['intens'].detach())
            for k in int_acc: records[k] = int_acc[k]
            iter_res.append(int_acc['ints_ratio'])

        if 'normal' in pred:
            n_acc, error_map = eval_utils.cal_normal_acc(data['normal'].detach(), pred['normal'].detach(), data['mask'].detach())
            for k in n_acc: records[k] = n_acc[k]
            iter_res.append(n_acc['n_err_mean'])
        
        return records, iter_res

    def prepare_visual(self):
        data, pred = self.data, self.pred
        visuals = [data['img'].detach(), data['mask'].detach(), (data['normal'].detach()+1)/2]

        if 'normal' in pred:
            _, error_map = eval_utils.cal_normal_acc(data['normal'].detach(), pred['normal'].detach(), data['mask'].detach())
            pred_n = (pred['normal'].detach() + 1) / 2
            masked_pred = pred_n * data['mask'].detach().expand_as(pred['normal'].detach())
            visuals += [masked_pred, error_map['angular_map']]

        if 'shading' in data:
            visuals.append(data['shading'].narrow(1, 0, 6))

        if 'shading' in pred:
            visuals.append(pred['shading'].narrow(1, 0, 6))
        return visuals

    def save_visual_detail(self, log, split, epoch, obj_path, obj_names):
        save_dir = log.config_save_detail_dir(split, epoch)
        data, pred = self.data, self.pred
        if 'normal' in pred:
            normal = pred['normal'].detach() * data['mask'].detach().expand_as(pred['normal'].detach())
            log.save_normal_mat(normal, obj_names, save_dir)

        if 'dirs' in pred and 'intens' in pred:
            gt_dirs = data['dirs'].view(-1, 3).detach().cpu()
            gt_intens = data['ints'].view(-1, 3).detach().cpu()
            log.plot_lighting(gt_dirs, gt_intens[:, 0], obj_names[0] + '_gt', save_dir)

            pred_dirs = pred['dirs'].view(-1, 3).detach().cpu()
            pred_intens = pred['intens'].view(-1, 3).detach().cpu()
            log.plot_lighting(pred_dirs, pred_intens[:, 0], obj_names[0] + '_est', save_dir)

            if self.opt['save_light']:
                log.save_light(pred_dirs, obj_path)  
                log.save_intens(pred_intens, obj_path) 
