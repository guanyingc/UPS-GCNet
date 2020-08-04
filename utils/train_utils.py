import torch
from models import model_utils
from utils import time_utils

def train(args, log, loader, model, epoch, recorder):
    model.train()
    log.print_write('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        model.parse_data(sample) 
        pred = model.forward(); 
        timer.update_time('Forward')

        model.optimize_weights()
        timer.update_time('Backward')
        
        loss = model.get_loss_terms()
        recorder.udpate_iter('train', loss.keys(), loss.values())

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.print_iters_summary(opt)

        if iters % args.train_save == 0:
            records, _ = model.prepare_records()
            visuals = model.prepare_visual() 
            recorder.udpate_iter('train', records.keys(), records.values())
            nrow = min(args.batch, 32)
            log.save_img_results(visuals, 'train', epoch, iters, nrow=nrow)
            log.plot_curves(recorder, 'train', epoch=epoch, intv=args.train_disp)

        if args.max_train_iter > 0 and iters >= args.max_train_iter: break
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.print_epoch_summary(opt)
