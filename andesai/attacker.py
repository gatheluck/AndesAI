import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import namedtuple

from andesai.attacks.pgd_attack import PGDAttack
from andesai.attacks.fw_attack import FrankWolfeAttack
from andesai.attacks.elastic_attack import ElasticAttack

class AttackerBuilder():
    AC = namedtuple('AttackConfig', ['norms'])
    ATTACK_CONFIG = {
        'pgd':          AC(norms=['linf', 'l2']),
        'fw':           AC(norms=['l1']),
        'cw':           AC(norms=['l2']),
        'deepfool':     AC(norms=['l1 l2']),
        'smoothfool':   AC(norms=['l1 l2']),
        'elastic':      AC(norms=['linf']),
    }

    def __init__(self):
        pass

    def __call__(self, at:str, at_norm:str, at_eps:float, nb_its:int, step_size:float, dataset:str, **kwargs):
        assert at_eps > 0.0
        assert nb_its > 0
        assert step_size > 0.0

        # check invalid attack name and norm
        if at+'-'+at_norm not in self.attacks_with_norm: 
            raise ValueError()

        if   at == 'pgd':
            attacker = PGDAttack(dataset=dataset, nb_its=nb_its, eps_max=at_eps, step_size=step_size, norm=at_norm)
        elif at == 'fw':
            attacker = FrankWolfeAttack(dataset=dataset, nb_its=nb_its, eps_max=at_eps)
        elif at == 'elastic':
            attacker = ElasticAttack(dataset=dataset, nb_its=nb_its, eps_max=eps_max, step_size=step_size)
        else:
            raise NotImplementedError

        return attacker

    @property
    def attacks_with_norm(self):
        return [attack+'-'+norm for attack in self.ATTACK_CONFIG.keys()
                                for norm in self.ATTACK_CONFIG[attack].norms]
    
    @property
    def attacks(self):
        return [attack for attack in self.ATTACK_CONFIG.keys()]

if __name__ == "__main__":
    import torch
    import torchvision
    from andesai.data import DatasetBuilder
    from andesai.model import ModelBuilder

    dataset_builder = DatasetBuilder(name='cifar10', root_path='../data')
    dataset = dataset_builder(train=True, normalize=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)

    model = model_builder = ModelBuilder(10, pretrained=False)['resnet18'].cuda()
    
    flags = {
        'dataset': 'cifar10',
        'at': 'fw',
        'at_norm': 'l1',
        'at_eps': 8.0,
        'nb_its': 10,
        'step_size': 0.8, 
    }

    attacker = AttackerBuilder()(**flags)

    for i, (x,t) in enumerate(loader):
        x = x.to('cuda', non_blocking=True)
        t = t.to('cuda', non_blocking=True)

        # adversarial attack
        if flags['at'] and flags['at_eps']>0:
            model.eval()
            model.zero_grad()
            x = attacker(model, x.detach(), t.detach())
            raise NotImplementedError


    
    
