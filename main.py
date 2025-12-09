import os, random
import numpy as np
import torch
import argparse
import wandb

from train import train

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parsing_args(c):
    data_path_default = getattr(c, 'data_path', None)
    workers_default = getattr(c, 'workers', 4)
    parser = argparse.ArgumentParser(description='msflow')
    parser.add_argument('--dataset', default='mvtec', type=str, 
                        choices=['mvtec', 'visa', 'shanghaitech', 'rail', 'rail_txt'], help='dataset name')
    parser.add_argument('--mode', default='train', type=str, 
                        help='train or test.')
    parser.add_argument('--amp_enable', action='store_true', default=False, 
                        help='use amp or not.')
    parser.add_argument('--wandb_enable', action='store_true', default=False, 
                        help='use wandb for result logging or not.')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='resume training or not.')
    parser.add_argument('--eval_ckpt', default='', type=str, 
                        help='checkpoint path for evaluation.')
    parser.add_argument('--class-names', default=['all'], type=str, nargs='+', 
                        help='class names for training')
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--batch-size', default=8, type=int, 
                        help='train batch size')
    parser.add_argument('--workers', default=workers_default, type=int,
                        help='dataloader workers (override default)')
    parser.add_argument('--data-path', default=data_path_default, type=str,
                        help='dataset root path (override default)')
    parser.add_argument('--train-list', default=None, type=str,
                        help='train list path (for rail_txt)')
    parser.add_argument('--test-list', default=None, type=str,
                        help='test list path (for rail_txt)')
    parser.add_argument('--train-fraction', default=1.0, type=float,
                        help='use first fraction of training data (0<frac<=1)')
    parser.add_argument('--test-limit', default=None, type=int,
                        help='limit number of test samples (use first N)')
    parser.add_argument('--meta-epochs', default=25, type=int,
                        help='number of meta epochs to train')
    parser.add_argument('--sub-epochs', default=4, type=int,
                        help='number of sub epochs to train')
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str, 
                        help='feature extractor')
    parser.add_argument('--pool-type', default='avg', type=str, 
                        help='pool type for extracted feature maps')
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, metavar='L', nargs='+',
                        help='number of flow blocks used in parallel flows.')
    parser.add_argument('--pro-eval', action='store_true', default=False, 
                        help='evaluate the pro score or not.')
    parser.add_argument('--pro-eval-interval', default=4, type=int, 
                        help='interval for pro evaluation.')

    args = parser.parse_args()

    for k, v in vars(args).items():
        setattr(c, k, v)
    
    # set data_path and class_names defaults if not provided
    if c.data_path is None:
        if c.dataset == 'mvtec':
            from datasets import MVTEC_CLASS_NAMES
            setattr(c, 'data_path', './data/MVTec')
            if c.class_names == ['all']:
                setattr(c, 'class_names', MVTEC_CLASS_NAMES)
        elif c.dataset == 'visa':
            from datasets import VISA_CLASS_NAMES
            setattr(c, 'data_path', './data/VisA_pytorch/1cls')
            if c.class_names == ['all']:
                setattr(c, 'class_names', VISA_CLASS_NAMES)
        elif c.dataset == 'shanghaitech':
            setattr(c, 'data_path', './data/shanghaitech')
            if c.class_names == ['all']:
                setattr(c, 'class_names', ['shanghaitech'])
        elif c.dataset == 'rail':
            setattr(c, 'data_path', './data/rail/rail_uvad_dataset')
            if c.class_names == ['all']:
                setattr(c, 'class_names', ['rail'])
        elif c.dataset == 'rail_txt':
            setattr(c, 'data_path', './data/rail/rail_uvad_dataset')
            if c.class_names == ['all']:
                setattr(c, 'class_names', ['rail'])
            if c.train_list is None:
                setattr(c, 'train_list', './train.txt')
            if c.test_list is None:
                setattr(c, 'test_list', './test.txt')
    else:
        # custom data_path; ensure class_names default for rail
        if c.dataset in ['rail', 'rail_txt'] and c.class_names == ['all']:
            setattr(c, 'class_names', ['rail'])

    if c.dataset in ['shanghaitech', 'rail', 'rail_txt']:
        c.input_size = (256, 256)
    else:
        c.input_size = (256, 256) if c.class_name == 'transistor' else (512, 512)

    return c

def main(c):
    c = parsing_args(c)
    init_seeds(seed=c.seed)
    c.version_name = 'msflow_{}_{}pool_pl{}'.format(c.extractor, c.pool_type, "".join([str(x) for x in c.parallel_blocks]))
    print(c.class_names)
    for class_name in c.class_names:
        c.class_name = class_name
        if c.dataset == 'mvtec' and c.class_name == 'transistor':
            c.input_size = (256, 256)
        elif c.dataset in ['shanghaitech', 'rail', 'rail_txt']:
            c.input_size = (256, 256)
        else:
            c.input_size = (512, 512)
        print('-+'*5, class_name, '+-'*5)
        c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.dataset, c.class_name)
        train(c)

if __name__ == '__main__':
    import default as c
    main(c)
