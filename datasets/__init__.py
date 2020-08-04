import importlib
import torch.utils.data

def find_dataset_from_string(dataset_name):
    datasetlib = importlib.import_module('datasets.%s' % dataset_name)
    class_name = dataset_name.replace('_', '')
    dataset = None
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == class_name.lower():
            dataset = cls
            break
    if dataset is None:
        print('In %s.py, there should be a class named %s' % (dataset_name, class_name))
        exit(0)
    return dataset

def custom_dataloader(args, log):
    log.print_write("=> fetching img pairs in %s" % (args.data_dir))
    train_set = find_dataset_from_string(args.dataset)(args, args.data_dir, 'train')
    val_set = find_dataset_from_string(args.dataset)(args, args.data_dir, 'val')

    if args.concat_data:
        log.print_write('****** Using cocnat data ******')
        log.print_write("=> fetching img pairs in '{}'".format(args.data_dir2))
        train_set2 = find_dataset_from_string(args.dataset)(args, args.data_dir2, 'train')
        val_set2 = find_dataset_from_string(args.dataset)(args, args.data_dir2, 'val')
        train_set  = torch.utils.data.ConcatDataset([train_set, train_set2])
        val_set    = torch.utils.data.ConcatDataset([val_set,   val_set2])

    log.print_write('Found Data:\t %d Train and %d Val' % (len(train_set), len(val_set)))
    log.print_write('\t Train Batch: %d, Val Batch: %d' % (args.batch, args.val_batch))

    use_gpu = len(args.gpu_ids) > 0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
        num_workers=args.workers, pin_memory=use_gpu, shuffle=True)
    test_loader   = torch.utils.data.DataLoader(val_set , batch_size=args.val_batch,
        num_workers=args.workers, pin_memory=use_gpu, shuffle=False)
    return train_loader, test_loader

def benchmark_loader(args, log):
    log.print_write("=> fetching img pairs in 'data/%s'" % (args.benchmark))
    test_set = find_dataset_from_string(args.benchmark)(args, 'test')

    log.print_write('Found Benchmark Data: %d samples' % (len(test_set)))
    log.print_write('\t Test Batch %d' % (args.test_batch))

    use_gpu = len(args.gpu_ids) > 0
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch,
        num_workers=args.workers, pin_memory=use_gpu, shuffle=False)
    return test_loader
