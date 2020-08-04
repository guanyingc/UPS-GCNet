import os
import torch
import importlib
from abc import ABC, abstractmethod
from . import model_utils

class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt['is_train']
        self.gpu_ids = opt['gpu_ids']
        self.device = torch.device('cuda:%d' % (self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.schedulers = []
        self.net_names = []
        self.fixed_net_names = []
        self.loss_terms = {}
        self.optimizers = []

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def prepare_inputs(self, data):
        pass

    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def optimize_weights(self):
        pass

    @abstractmethod
    def prepare_visual(self):
        pass

    @abstractmethod
    def prepare_records(self):
        pass

    def parse_data(self, sample):
        data = {}
        n, c, h, w = sample['img'].shape
        data['img'] = sample['img'].to(self.device)
        data['normal'] = sample['normal'].to(self.device)
        data['mask'] = sample['mask'].to(self.device)

        ints_reshape = torch.stack(torch.split(sample['ints'], 3, 1), 1).view(n, -1, 3)
        data['ints'] = ints_reshape.to(self.device)

        dirs_reshape = torch.stack(torch.split(sample['dirs'], 3, 1), 1).view(n, -1, 3)
        data['dirs'] = dirs_reshape.to(self.device)

        if self.opt['est_sd']:
            data['shading'] = sample['shading'].to(self.device)
        if self.opt['est_spec_sd']:
            data['spec_sd'] = sample['spec_sd'].to(self.device)
        if self.opt['est_shadow']:
            data['shadow'] = sample['shadow'].to(self.device)
        self.data = data
        return data

    def get_loss_terms(self):
        return self.loss_terms

    def print_networks(self, log=None):
        for name in self.net_names + self.fixed_net_names:
            network = getattr(self, name)
            print(network)
            log.print_write("=> Parameters of %s: %d" % (name, model_utils.get_params_num(network)))

    def import_network(self, net_name):
        network_file = importlib.import_module('models.archs.%s' % net_name)
        network = getattr(network_file, net_name)
        return network

    def eval(self):
        for name in self.net_names + self.fixed_net_names:
            net = getattr(self, name)
            net.eval()

    def train(self):
        for name in self.net_names + self.fixed_net_names:
            net = getattr(self, name)
            net.train()
    
    def setup_lr_scheduler(self):
        self.schedulers = [model_utils.get_lr_scheduler(self.opt, optimizer) for optimizer in self.optimizers]

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def save_checkpoint(self, epoch, records=None):
        for name in self.net_names:
            network = getattr(self, name)
            if len(self.gpu_ids) > 0:
                state = {'state_dict': network.module.state_dict()}
            else:
                state = {'state_dict': network.state_dict()}
            torch.save(state, os.path.join(self.opt['cp_dir'], '%s_checkp_%d.pth' % (name, epoch)))

    def load_checkpoint(self, log):
        for name in self.net_names + self.fixed_net_names:
            checkp_path = self.opt['%s_checkp' % name]
            if checkp_path != '':
                log.print_write("==> [%s] loading pre-trained model from %s" % (name, checkp_path))
                network = getattr(self, name)
                self._load_checkpoint(network, checkp_path, log) 

    def _load_checkpoint(self, network, checkp_path, log):
        if isinstance(network, torch.nn.DataParallel):
            network = network.module
        checkpoint = torch.load(checkp_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        network.load_state_dict(state_dict, strict=self.opt['strict'])

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
