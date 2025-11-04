import argparse
from pathlib import Path
from ast import arg

def parser_args():
    # available_models = ['Base-R101-448', 'Base-R101-576']
    parser = argparse.ArgumentParser(description='Visual Prompt Tuning for Multi-label Learning Training')
    
    parser.add_argument('--output', metavar='DIR', help='path to output folder')


    # train
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-p', '--print-freq', default=400, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpus', default='0,1', help='select GPUS (default: 0)')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')


    # distribution training # --world-size -1 --rank -1
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=-1, type=int, 
                        help='local rank for DistributedDataParallel')

    # config file
    parser.add_argument('--cfg', type=str, required=True, default='./config/voc2007.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    args = parser.parse_args()

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    return args

# parser.add_argument('--dataname', help='dataname', default='coco14', choices=['coco14', 'voc2007', 'voc2012', 'nus_wide', 'nuswide', 'vg500', 'coco14_csl', 'voc2007partial', 'coco14partial'])