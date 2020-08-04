from collections import OrderedDict
import numpy as np

class Records(object):
    """
    Records->Train,Val->Loss,Accuracy->Epoch1,2,3->[v1,v2]
    IterRecords->Train,Val->Loss, Accuracy,->[v1,v2]
    """
    def __init__(self, records=None):
        if records == None:
            self.records = OrderedDict()
        else:
            self.records = records
        self.iter_rec = OrderedDict()
        self.classes = ['loss', 'acc', 'err', 'ratio']

    def reset_iter(self):
        self.iter_rec.clear()

    def check_dict(self, a_dict, key, sub_type='dict'):
        if key not in a_dict.keys():
            if sub_type == 'dict':
                a_dict[key] = OrderedDict()
            if sub_type == 'list':
                a_dict[key] = []

    def udpate_iter(self, split, keys, values):
        self.check_dict(self.iter_rec, split, 'dict')
        for k, v in zip(keys, values):
            self.check_dict(self.iter_rec[split], k, 'list')
            self.iter_rec[split][k].append(v)

    def save_iter_record(self, epoch, reset=True):
        for s in self.iter_rec.keys(): # s stands for split
            self.check_dict(self.records, s, 'dict')
            for k in self.iter_rec[s].keys():
                self.check_dict(self.records[s], k, 'dict')
                self.check_dict(self.records[s][k], epoch, 'list')
                self.records[s][k][epoch].append(np.mean(self.iter_rec[s][k]))
        if reset: 
            self.reset_iter()

    def insert_record(self, split, key, epoch, value):
        self.check_dict(self.records, split, 'dict')
        self.check_dict(self.records[split], key, 'dict')
        self.check_dict(self.records[split][key], epoch, 'list')
        self.records[split][key][epoch].append(value)

    def iter_rec_to_string(self, split, epoch):
        rec_strs = ''
        for c in self.classes:
            strs = ''
            for k in self.iter_rec[split].keys():
                if (c in k.lower()):
                    strs += '{}: {:.3f}| '.format(k, np.mean(self.iter_rec[split][k]))
            if strs != '':
                rec_strs += '\t [{}] {}\n'.format(c.upper(), strs)
        self.save_iter_record(epoch)
        return rec_strs

    def epoch_rec_to_string(self, split, epoch):
        rec_strs = ''
        for c in self.classes:
            strs = ''
            for k in self.records[split].keys():
                if (c in k.lower()) and (epoch in self.records[split][k].keys()):
                    strs += '{}: {:.3f}| '.format(k, np.mean(self.records[split][k][epoch]))
            if strs != '':
                rec_strs += '\t [{}] {}\n'.format(c.upper(), strs)
        return rec_strs

    def record_to_dict_of_array(self, splits, epoch=-1, intv=1):
        if len(self.records) == 0: return {}
        if type(splits) == str: splits = [splits]

        dict_of_array = OrderedDict()
        for split in splits:
            for k in self.records[split].keys():
                y_array, x_array = [], []
                if epoch < 0:
                    for ep in self.records[split][k].keys():
                        y_array.append(np.mean(self.records[split][k][ep]))
                        x_array.append(ep)
                else:
                    if epoch in self.records[split][k].keys():
                        y_array = np.array(self.records[split][k][epoch])
                        x_array = np.linspace(intv, intv*len(y_array), len(y_array))
                dict_of_array[split[0] + split[-1] + '_' + k]      = y_array
                dict_of_array[split[0] + split[-1] + '_' + k+'_x'] = x_array
        return dict_of_array
