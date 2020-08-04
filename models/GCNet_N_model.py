import torch
from .base_model import BaseModel
from .GCNet_model import GCNetModel
from utils import eval_utils
from . import model_utils
from collections import OrderedDict

class GCNetNModel(GCNetModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--GCNet_name', default='GCNet')
        parser.add_argument('--GCNet_checkp', default='')
        parser.add_argument('--Normal_Net_name', default='PS_FCN')
        parser.add_argument('--Normal_Net_checkp', default='')
        parser.add_argument('--in_est_n', default=True, action='store_false')
        parser.add_argument('--in_est_sd', default=True, action='store_false')
        parser.set_defaults(test_resc=False)

        if is_train:
            parser.set_defaults(milestones=[2, 4, 6, 8, 10], epochs=20, init_lr=0.0005) 
        else:
            parser.set_defaults(test_root='GCNet_checkp') 
            # In run_model mode, results will be save in the same directory as args.GCNet_checkp

        str_keys = ['GCNet_name', 'Normal_Net_name']
        val_keys = [ ]
        bool_keys = ['in_est_n', 'in_est_sd'] 
        return parser, str_keys, val_keys, bool_keys

    def __init__(self, args, log):
        opt = vars(args)
        BaseModel.__init__(self, opt)
        self.fixed_net_names = ['GCNet']
        self.net_names = ['Normal_Net']

        c_in = self.get_in_channel_nums(opt, log)
        self.GCNet = self.import_network(args.GCNet_name)(opt, c_in)
        self.GCNet = model_utils.init_net(self.GCNet, gpu_ids=args.gpu_ids)
        self.set_requires_grad(self.GCNet, requires_grad=False)

        self.Normal_Net = self.import_network(args.Normal_Net_name)(opt, 6)
        self.Normal_Net = model_utils.init_net(self.Normal_Net, gpu_ids=args.gpu_ids)

        if self.is_train: # Criterion
            self.normal_crit = model_utils.NormalCrit(opt, log)
            self.optimizer = torch.optim.Adam(self.Normal_Net.parameters(), lr=args.init_lr, betas=(args.beta_1, args.beta_2))
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
        self.GCNet_pred = self.GCNet(self.data)
        self.Normal_Net_input = self.prepare_NNet_inputs(self.data, self.GCNet_pred)
        self.pred = self.Normal_Net(self.Normal_Net_input)
        for k in self.GCNet_pred.keys():
            self.pred[k] = self.GCNet_pred[k]
        return self.pred

    def optimize_weights(self):
        normal_loss, normal_loss_term = self.normal_crit(self.pred['normal'], self.data['normal'])
        self.loss = self.opt['normal_w'] * normal_loss
            
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.loss_terms = {}
        for k in normal_loss_term: self.loss_terms[k] = normal_loss_term[k]
    
    def prepare_records(self):
        records, iter_res = OrderedDict(), []
        records_GCNet, iter_res_GCNet = GCNetModel.prepare_records(self)
        records.update(records_GCNet)
        iter_res += iter_res_GCNet
        return records, iter_res

    def prepare_visual(self):
        visuals = []
        visuals += GCNetModel.prepare_visual(self)
        #data, pred = self.data, self.pred
        #if self.opt['s2_est_n']:
        #    _, error_map = eval_utils.cal_normal_acc(data['normal'].detach(), pred['normal'].detach(), data['mask'].detach())
        #    pred_n = (pred['normal'].detach() + 1) / 2
        #    masked_pred = pred_n * data['mask'].detach().expand_as(pred['normal'].detach())
        #    visuals += [masked_pred, error_map['angular_map']]
        return visuals

    def save_visual_detail(self, log, split, epoch, obj_path, obj_names):
        GCNetModel.save_visual_detail(self, log, split, epoch, obj_path, obj_names)
        save_dir = log.config_save_detail_dir(split, epoch)
        #data, pred = self.data, self.pred
        #if self.opt['s2_est_n']:
        #    normal = pred['normal'].data * data['mask'].data.expand_as(pred['normal'].data)
        #    log.save_normal_mat(normal, obj_names, save_dir)
