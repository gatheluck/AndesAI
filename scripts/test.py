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
# optional transform
@click.option('--num_divide', type=int, default=0, help='number of patches which is used in PatchShuffle')

def main(**kwargs):
    test(**kwargs)

def test(**kwargs):
    """
    test model on specific cost and specific adversarial perturbation.
    """
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    assert FLAGS.nb_its>=0
    assert FLAGS.attack_eps>=0
    assert FLAGS.num_divide>=0
    if FLAGS.attack_eps>0 and FLAGS.num_divide>0:
        raise ValueError('Adversarial Attack and Patch Shuffle should not be used at same time')

    # optional transform
    optional_transform=[]
    optional_transform.extend([PatchSuffle(FLAGS.num_divide)] if FLAGS.num_divide else [])

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