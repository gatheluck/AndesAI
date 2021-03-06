import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import re
import click
import uuid
import glob

from collections import OrderedDict

from external.dada.flag_holder import FlagHolder
from external.dada.logger import Logger
from scripts.test import test


# options
@click.command()
# target
@click.option('-t', '--target_dir', type=str, required=True)
# model
@click.option('-a', '--arch', type=str, required=True)
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-N', '--batch_size', type=int, default=1024)
# Fourier noise
@click.option('--fn_eps', type=float, required=True, help='perturbation size of Fourier Noise')
@click.option('--fn_max_index_h', type=int, default=16, help='maximum hight index of Fourier Noise')
@click.option('--fn_max_index_w', type=int, default=16, help='maximum width index of Fourier Noise')
# log
@click.option('-s', '--suffix', type=str, default='')

def main(**kwargs):
    test_fourier(**kwargs)

def parse_weight_basename(weight_basename):
    ret_dict = dict()

    # remove ext
    basename, ext = os.path.splitext(weight_basename)

    # extract coverage and else
    # \d : any single number
    # \w : any single number or charactor
    # .  : any single charactor
    # +  : sequence more than one time
    # *  : sequence more than zero time

    # 'weight_final_coverage_{something}'
    pattern = r'weight_final_(.*)'
    result = re.match(pattern, basename)

    at_info = result.group(1)

    if at_info == 'std':
        ret_dict['at'] = None
        ret_dict['at_norm'] = None
        ret_dict['at_eps'] = 0.0
    else:
        # '{at}-{at_norm}_eps-{at_eps}' 
        pattern = r'(\w+)-(\w+)_eps-(\d+)'
        result = re.match(pattern, at_info)

        ret_dict['at'] = result.group(1)
        ret_dict['at_norm'] = result.group(2)
        ret_dict['at_eps'] = float(result.group(3))
        
    return ret_dict

def test_fourier(**kwargs):
    """
    this script loads all 'weight_final_{something}.pth' files which exisits under 'kwargs.target_dir' and execute test.
    if there is exactly same file, the result becomes the mean of them.
    the results are saved as csv file.

    'target_dir' should be like follow
    (.pth file name should be "weight_final_coverage_{}") 

    ~/target_dir/XXXX/weight_final_pgd-linf_eps-0.pth
                     ...
                     /weight_final_pgd-linf_eps-8.pth
                     /weight_final_pgd-linf_eps-16.pth
                     ...
                /YYYY/weight_final_pgd-linf_eps-0.pth
                     ...
                     /weight_final_pgd-linf_eps-8.pth
                     /weight_final_pgd-linf_eps-16.pth
                     ...
    """
    # flags
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # paths
    run_dir  = '../scripts'
    if os.path.splitext(FLAGS.target_dir)[-1]!='.pth':
        target_path = os.path.join(FLAGS.target_dir, '**/weight_final*.pth')
        weight_paths = sorted(glob.glob(target_path, recursive=True), key=lambda x: os.path.basename(x))
        log_path = os.path.join(FLAGS.target_dir, 'test{}.csv'.format(FLAGS.suffix))
    else:
        weight_paths = [FLAGS.target_dir]
        log_path = os.path.join(os.path.dirname(FLAGS.target_dir), 'test{}.csv'.format(FLAGS.suffix))

    # logging
    logger = Logger(path=log_path, mode='test', use_wandb=False, flags=FLAGS)

    for weight_path in weight_paths:
        for index_h in range(-FLAGS.fn_max_index_h, FLAGS.fn_max_index_h+1):
            for index_w in range(-FLAGS.fn_max_index_w, FLAGS.fn_max_index_w+1):
                # continue when indices are 0
                if index_h==0 or index_w==0: continue

                # parse basename
                basename = os.path.basename(weight_path)
                ret_dict = parse_weight_basename(basename)

                # keyword args for test function
                # variable args
                kw_args = {}
                kw_args['arch'] = FLAGS.arch
                kw_args['weight'] = weight_path
                kw_args['dataset'] = FLAGS.dataset
                kw_args['dataroot'] = FLAGS.dataroot
                kw_args['batch_size'] = FLAGS.batch_size
                kw_args['fn_eps'] = FLAGS.fn_eps
                kw_args['fn_index_h'] = index_h
                kw_args['fn_index_w'] = index_w

                # default args
                kw_args['num_workers'] = 8
                kw_args['normalize'] = True
                
                # run test
                out_dict = test(**kw_args)

                metric_dict = OrderedDict()
                # model
                metric_dict['arch'] = FLAGS.arch
                # Fourier noise
                metric_dict['fn_eps'] = FLAGS.fn_eps
                metric_dict['fn_index_h'] = index_h
                metric_dict['fn_index_w'] = index_w
                # at
                metric_dict['at'] = ret_dict['at']
                metric_dict['at_norm'] = ret_dict['at_norm']
                metric_dict['at_eps'] = ret_dict['at_eps']
                # path
                metric_dict['path'] = weight_path
                metric_dict.update(out_dict)

                # log
                logger.log(metric_dict)

if __name__ == '__main__':
    main()