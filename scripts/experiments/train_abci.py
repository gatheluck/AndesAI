import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import subprocess
import click
import uuid

from collections import namedtuple

from external.dada.flag_holder import FlagHolder
from external.abci_util.script_generator import generate_script
from andesai.attacker import AttackerBuilder

TP = namedtuple('TrainingParameter', 'num_epochs batch_size ')
# TRAINING_PARAMS = {
#     'cifar10':      {'num_epochs':, 'batch_size':, }
#     'imagenet100':  {'num_epochs':, 'batch_size':, }
# }


ATTACK_CONFIG = AttackerBuilder.ATTACK_CONFIG
AT_PARAMS = {
    'pgd-linf': {
        'nb_its': 10,
        'at_eps': {
            'cifar10':    [1, 2, 4, 8, 16, 32],
            'imagenet100':[1, 2, 4, 8, 16, 32],
        },
    },
    'pgd-l2': {
        'nb_its': 10,
        'at_eps': {
            'cifar10':    [40,  80,  160, 320,  640,  2560],
            'imagenet100':[150, 300, 600, 1200, 2400, 4800],
        },
    },
    'fw-l1': {
        'nb_its': 10,
        'at_eps': {
            'cifar10':    [195,    390,   780,   1560,   6240,   24960],
            'imagenet100':[9562.5, 19125, 76500, 153000, 306000, 612000],
        },
    },
    'elastic-linf': {
        'nb_its': 30,
        'at_eps': {
            'cifar10':    [0.125, 0.25, 0.5, 1, 2, 8],
            'imagenet100':[0.25,  0.5,  2,   4, 8, 16],
        },
    },
    # 'snow-linf': {
    #     'nb_its': 30,
    #     'at_eps': {
    #         'cifar10':[0, ],
    #         'imagenet100':[0.062, 0.125, 0.250, 2, 4, 8],
    #     },
    # },
}

# options
@click.command()
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
# optimization
@click.option('--num_epochs', type=int, default=200)
@click.option('-N', '--batch_size', type=int, default=1024)
# logging
@click.option('-l', '--log_dir', type=str, required=True)
@click.option('--ex_id', type=str, default=uuid.uuid4().hex, help='id of the experiments')
# selective loss
@click.option('--cost', type=float, default=None)
# at
@click.option('--nb_its', type=int, default=10)
# option for abci
@click.option('--script_root', type=str, required=True)
@click.option('--run_dir', type=str, required=True)
@click.option('--abci_log_dir', type=str, default='~/abci_log')
@click.option('--user', type=str, required=True)
@click.option('--env', type=str, required=True)

def main(**kwargs):
    train_multi(**kwargs)

def train_multi(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # create script output dir
    script_dir = os.path.join(FLAGS.script_root, FLAGS.ex_id)
    os.makedirs(script_dir, exist_ok=True)

    costs = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70] if not FLAGS.cost else [FLAGS.cost]
    ats = ['pgd']
    at_norms = ['linf', 'l2']

    EPS = {
        'pgd-linf': [0, 1, 2, 4, 8, 16],
        'pgd-l2':   [0, 40, 80, 160, 320, 640],
    }

    for cost in sorted(costs):
        for at in ats:
            for at_norm in at_norms:
                key = at+'-'+at_norm
                for at_eps in EPS[key]:

                    suffix = '_cost-{cost:0.2f}_{at}-{at_norm}_eps-{at_eps:d}'.format(
                        cost=cost, at=at, at_norm=at_norm, at_eps=at_eps) 

                    log_dir = os.path.join(FLAGS.log_dir, FLAGS.ex_id)
                    os.makedirs(log_dir, exist_ok=True)

                    cmd = 'python train.py \
                          -d {dataset} \
                          --dataroot {dataroot} \
                          --num_epochs {num_epochs} \
                          --batch_size {batch_size} \
                          --cost {cost} \
                          --at {at} \
                          --nb_its {nb_its} \
                          --at_eps {at_eps} \
                          --at_norm {at_norm} \
                          -s {suffix} \
                          -l {log_dir}'.format(
                            dataset=FLAGS.dataset,
                            dataroot=FLAGS.dataroot,
                            num_epochs=FLAGS.num_epochs,
                            batch_size=FLAGS.batch_size,
                            cost=cost,
                            at=at,
                            nb_its=FLAGS.nb_its,
                            at_eps=at_eps,
                            at_norm=at_norm,
                            suffix=suffix,
                            log_dir=log_dir)

                    script_basename = suffix.lstrip('_')+'.sh'
                    script_path = os.path.join(script_dir, script_basename)
                    generate_script(cmd, script_path, FLAGS.run_dir, FLAGS.abci_log_dir, FLAGS.ex_id, FLAGS.user, FLAGS.env)

if __name__ == '__main__':
    main()