import os
import torch
from models import model_utils
from utils import eval_utils, time_utils 
import numpy as np

def get_itervals(args, split):
    if split not in ['train', 'val', 'test']:
        split = 'test'
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    stop_iters = args_var['max_'+split+'_iter']
    return disp_intv, save_intv, stop_iters

def test(args, log, split, loader, model, epoch, recorder):
    model.eval()
    log.print_write('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    with torch.no_grad():
        for i, sample in enumerate(loader):

            data = model.parse_data(sample) 
            pred = model.forward(); 
            timer.update_time('Forward')

            loss = model.get_loss_terms()
            if loss != None: 
                recorder.udpate_iter(split, loss.keys(), loss.values())

            records, iter_res = model.prepare_records()
            recorder.udpate_iter(split, records.keys(), records.values())

            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.print_iters_summary(opt)

            if iters % save_intv == 0:
                visuals = model.prepare_visual() 

                nrow = min(data['img'].shape[0], 32)
                log.save_img_results(visuals, split, epoch, iters, nrow=nrow)
                log.plot_curves(recorder, split, epoch=epoch, intv=disp_intv)
                                
                if hasattr(args, 'save_detail') and args.save_detail or (split == 'test'):
                    model.save_visual_detail(log, split, epoch, sample['path'], sample['obj'])

            if stop_iters > 0 and iters >= stop_iters: break
    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.print_epoch_summary(opt)
