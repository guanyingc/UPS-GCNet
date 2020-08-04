import torch
from .base_model import BaseModel
from .L_model import LModel
from utils import eval_utils
from . import model_utils
from collections import OrderedDict

class LNModel(LModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--L_Net_name',  default='L_Net')
        parser.add_argument('--L_Net_checkp', default='')
        parser.add_argument('--N_Net_name',  default='N_Net')
        parser.add_argument('--N_Net_checkp', default='')
        parser.set_defaults(test_resc=False)
        if is_train:
            parser.set_defaults(milestones=[2, 4], epochs=5, init_lr=0.0002) 
        else:
            parser.set_defaults(test_root='L_Net_checkp') 
            # In run_model mode, results will be save in the same directory as args.L_Net_checkp
        str_keys = ['L_Net_name', 'N_Net_name']
        val_keys = [ ]
        bool_keys = [] 
        return parser, str_keys, val_keys, bool_keys

    def __init__(self, args, log):
        opt = vars(args)
        BaseModel.__init__(self, opt)
        self.fixed_net_names = ['L_Net'] # The network weight are fixed during
        self.net_names = ['N_Net']

        c_in = LModel.get_in_channel_nums(self, opt, log)
        self.L_Net = self.import_network(args.L_Net_name)(opt, c_in)
        self.L_Net = model_utils.init_net(self.L_Net, gpu_ids=args.gpu_ids)
        self.set_requires_grad(self.L_Net, requires_grad=False)

        self.N_Net = self.import_network(args.N_Net_name)(opt, 6)
        self.N_Net = model_utils.init_net(self.N_Net, gpu_ids=args.gpu_ids)

        if self.is_train: # Criterion
            self.normal_crit = model_utils.NormalCrit(opt, log)
            self.optimizer = torch.optim.Adam(self.N_Net.parameters(), lr=args.init_lr, betas=(args.beta_1, args.beta_2))
            self.optimizers.append(self.optimizer)
            self.setup_lr_scheduler()

        self.load_checkpoint(log)

    def prepare_NNet_inputs(self, data, pred):
        imgs = torch.split(data['img'], 3, 1)
        dirs = torch.split(pred['dirs'], 1, 1)
        ints = torch.split(pred['intens'], 1, 1)
        
        inputs = []

        imgs_int_normalized = []
        n, c, h, w = imgs[0].shape
        for i in range(len(imgs)):
            l_int = torch.diag(1.0 / (ints[i].contiguous().view(-1) + 1e-8))
            img = imgs[i].contiguous().view(n * c, h * w)
            img = torch.mm(l_int, img).view(n, c, h, w)
            imgs_int_normalized.append(img)

        imgs = imgs_int_normalized

        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            l_dir = dirs[i] if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1)
            img_light = torch.cat([imgs[i], l_dir.expand_as(imgs[i])], 1)
            inputs.append(img_light)
        return inputs

    def forward(self):
        self.L_Net_inputs = LModel.prepare_inputs(self, self.data) 
        self.L_Net_pred = self.L_Net(self.L_Net_inputs)

        self.N_Net_inputs = self.prepare_NNet_inputs(self.data, self.L_Net_pred)
        self.pred = self.N_Net(self.N_Net_inputs)

        for k in self.L_Net_pred.keys():
            self.pred[k] = self.L_Net_pred[k]
        return self.pred

    def optimize_weights(self):
        normal_loss, normal_loss_term = self.normal_crit(self.pred['normal'], self.data['normal'])
        self.loss = self.opt['normal_w'] * normal_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.loss_terms = {}
        for k in normal_loss_term: 
            self.loss_terms[k] = normal_loss_term[k]
    
    def prepare_records(self):
        records = OrderedDict()
        iter_res = []

        records_s1, iter_res_s1 = LModel.prepare_records(self)
        records.update(records_s1)
        iter_res += iter_res_s1

        #data, pred = self.data, self.pred
        #if self.opt['s2_est_n']:
        #    n_acc, error_map = eval_utils.cal_normal_acc(data['normal'].detach(), pred['normal'].detach(), data['mask'].detach())
        #    for k in n_acc: records[k] = n_acc[k]
        #    iter_res.append(n_acc['n_err_mean'])
        return records, iter_res

    def prepare_visual(self):
        visuals = []
        visuals += LModel.prepare_visual(self)

        #data, pred = self.data, self.pred
        #if self.opt['s2_est_n']:
        #    _, error_map = eval_utils.cal_normal_acc(data['normal'].detach(), pred['normal'].detach(), data['mask'].detach())
        #    pred_n = (pred['normal'].detach() + 1) / 2
        #    masked_pred = pred_n * data['mask'].detach().expand_as(pred['normal'].detach())
        #    visuals += [masked_pred, error_map['angular_map']]
        return visuals
    
    def save_visual_detail(self, log, split, epoch, obj_path, obj_names):
        LModel.save_visual_detail(self, log, split, epoch, obj_path, obj_names)
        #save_dir = log.config_save_detail_dir(split, epoch)
        #data, pred = self.data, self.pred
        #if self.opt['s2_est_n']:
        #    normal = pred['normal'].data * data['mask'].data.expand_as(pred['normal'].data)
        #    log.save_normal_mat(normal, obj_names, save_dir)
