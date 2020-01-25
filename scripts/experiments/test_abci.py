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
from andesai.config import AT_PARAMS

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
# at
@click.option('--nb_its', type=int, default=50)
# option for abci
@click.option('--script_root', type=str, required=True)
@click.option('--run_dir', type=str, required=True)
@click.option('--abci_log_dir', type=str, default='~/abci_log')
@click.option('--user', type=str, required=True)
@click.option('--env', type=str, required=True)

def main(**kwargs):
    test_abci(**kwargs)

def test_abci**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # create script output dir
    script_dir = os.path.join(FLAGS.script_root, FLAGS.ex_id)
    os.makedirs(script_dir, exist_ok=True)


    # loop for attack method
    for attack_method in AttackerBuilder.ATTACK_CONFIG.keys():
        for attack_norm in AttackerBuilder.ATTACK_CONFIG[attack_method].norms:

            attack_method_with_norm = attack_method + '-' + attack_norm    
            for attack_eps in AT_PARAMS[attack_method_with_norm][FLAGS.dataset]:

                suffix = '_{attack_method}-{attack_norm}_eps-{attack_eps:d}'.format(
                            attack_method=attack_method, attack_norm=attack_norm, attack_eps=attack_eps) 

                log_dir = os.path.join(FLAGS.log_dir, FLAGS.ex_id)
                    os.makedirs(log_dir, exist_ok=True)

                    cmd = 'python experiments/test_abci.py \
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