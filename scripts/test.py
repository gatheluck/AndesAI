import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
from collections import OrderedDict

import torch
import torchvision

from external.dada.flag_holder import FlagHolder
from external.dada.metric import MetricDict
from external.dada.io import print_metric_dict
from external.dada.io import save_model
from external.dada.io import load_model
from external.dada.logger import Logger

from andesai.attacks.attacks import get_step_size
from andesai.model import ModelBuilder
from andesai.data import DatasetBuilder
from andesai.attacker import AttackerBuilder
from andesai.transforms.patch_shuffle import PatchSuffle
from andesai.transforms.fourier_noise import FourierNoise
from andesai.evaluator import Evaluator

# options
@click.command()
# model
@click.option('-a', '--arch', type=str, required=True)
# model
@click.option('-w', '--weight', type=str, required=True, help='model weight path')
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=1024)
@click.option('--normalize', is_flag=True, default=True)
# adversarial attack
@click.option('--attack', type=str, default=None)
@click.option('--attack_norm', type=str, default=None)
@click.option('--attack_eps', type=float, default=0.0)
@click.option('--nb_its', type=int, default=50)
@click.option('--step_size', type=float, default=None)
# optional transform (PatchShuffle)
@click.option('--ps_num_divide', type=int, default=0, help='number of patches which is used in PatchShuffle')
# optional transform (FourierNoise)
@click.option('--fn_eps', type=float, default=0.0, help='perturbation size of Fourier Noise')
@click.option('--fn_index_h', type=int, default=1, help='hight index of Fourier Noise')
@click.option('--fn_index_w', type=int, default=1, help='width index of Fourier Noise')


def main(**kwargs):
    test(**kwargs)

def set_default(**kwargs):
    """
    This funcition is redandant. Currently, function test is imported from other function e.g. test_fourier.py.
    At that case, argument which is specified by click is not defined in kwargs. so we set default value here.
    This problem might be soloved by using subprocess.  
    """
    DEFAULT_ARGS = {
        'attack':None,
        'attack_norm':None,
        'attack_eps':0.0,
        'nb_its':0,
        'step_size':None,
        'ps_num_divide':0,
        'fn_eps':0.0,
        'fn_index_h':1,
        'fn_index_w':1,
    }

    for key, val in DEFAULT_ARGS.items():
        if key not in kwargs.keys():
            kwargs[key] = val

    return kwargs

def test(**kwargs):
    """
    test model on specific cost and specific adversarial perturbation.
    """
    
    kwargs = set_default(**kwargs)

    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    assert FLAGS.nb_its>=0
    assert FLAGS.attack_eps>=0
    assert FLAGS.ps_num_divide>=0
    if FLAGS.attack_eps>0 and FLAGS.ps_num_divide>0:
        raise ValueError('Adversarial Attack and Patch Shuffle should not be used at same time')
    if FLAGS.attack_eps>0 and FLAGS.fn_eps>0:
        raise ValueError('Adversarial Attack and Fourier Noise should not be used at same time')
    if FLAGS.ps_num_divide>0 and FLAGS.fn_eps>0:
        raise ValueError('Patch Shuffle and Fourier Noise should not be used at same time')

    # optional transform
    optional_transform=[]
    optional_transform.extend([PatchSuffle(FLAGS.ps_num_divide)] if FLAGS.ps_num_divide else [])
    optional_transform.extend([FourierNoise(FLAGS.fn_index_h, FLAGS.fn_index_w, FLAGS.fn_eps)] if FLAGS.fn_eps else [])
    
    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)
    test_dataset   = dataset_builder(train=False, normalize=FLAGS.normalize, optional_transform=optional_transform)
    test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    # model (load from checkpoint) 
    num_classes = dataset_builder.num_classes
    model = ModelBuilder(num_classes=num_classes, pretrained=False)[FLAGS.arch].cuda()
    load_model(model, FLAGS.weight)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # adversarial attack
    if FLAGS.attack and FLAGS.attack_eps>0:    
        # get step_size
        step_size = get_step_size(FLAGS.attack_eps, FLAGS.nb_its) if not FLAGS.step_size else FLAGS.step_size
        FLAGS._dict['step_size'] = step_size
        assert step_size>=0

        # create attacker
        attacker = AttackerBuilder()(method=FLAGS.attack, norm=FLAGS.attack_norm, eps=FLAGS.attack_eps, **FLAGS._dict)
    
    # pre epoch misc
    test_metric_dict = MetricDict()

    # test
    for i, (x,t) in enumerate(test_loader):
        model.eval()
        x = x.to('cuda', non_blocking=True)
        t = t.to('cuda', non_blocking=True)
        loss_dict = OrderedDict()

        # adversarial samples
        if FLAGS.attack and FLAGS.attack_eps>0:
            # create adversarial sampels
            model.zero_grad()
            x = attacker(model, x.detach(), t.detach())

        with torch.autograd.no_grad():
            # forward
            model.zero_grad()
            logit = model(x)
            
            # compute selective loss
            ce_loss = torch.nn.CrossEntropyLoss()(logit, t).detach().cpu().item()
            loss_dict['loss'] = ce_loss

            # evaluation
            evaluator = Evaluator(logit.detach(), t.detach(), selection_out=None)
            loss_dict.update(evaluator())

        test_metric_dict.update(loss_dict)

    # post epoch
    print_metric_dict(None, None, test_metric_dict.avg, mode='test')

    return test_metric_dict.avg

if __name__ == '__main__':
    main()