import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import torch
import torchvision

from andesai.attacks.attacks import AttackWrapper

class DeepFoolAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, dataset, num_candidate_classes, norm='l2', metric='logit', rand_init=True, scale_each=False):
    """
    Parameters:
        dataset (str):         dataset name
        nb_its (int):          Number of PGD iterations.
        eps_max (float):       The max norm, in pixel space.
        step_size (float):     The max step size, in pixel space.
        rand_init (bool):      Whether to init randomly in the norm ball
        scale_each (bool):     Whether to scale eps for each image in a batch separately
    """
    super().__init__(dataset)
    assert norm in 'l1 l2'.split()
    assert metric in 'logit loss'.split()

    self.num_candidate_classes = num_candidate_classes
    self.norm = norm
    self.metric = metric

    def _init(self,)
        return candidate_indices

    def _forward(self, pixel_model, pixel_img, target, avoid_target=True):
        

        with torch.autograd.no_grad():
            candidate_classes = pixel_model(pixel_img).topk(num_candidate_classes, dim=1).indices # (B, #candidate)
            init_predicted_classes = candidate_indices[:,0] # (B)
            predicted_classes = init_predicted_classes # (B)
            
            pixel_result = pixel_img.clone()
            delta = torch.zeros_like(pixel_img).cuda()

        it=0
        while predicted_classes==init_predicted_classes and it<nb_its:
            x = pixel_result.clone()
            x.requires_grad_()

            pixel_model.zero_grad()
            
            logit = pixel_model(x) # (B, #class)
            if self.metric == 'logit':
                metric = logit
            elif self.metric == 'loss':
                raise NotImplementedError
            else:
                raise NotImplementedError


            w = torch.zeros_like(pixel_img).cuda()

            # range starts from 1 because 0 is init_prediction results.
            for it_class in range(1, self.num_candidate_classes):

        
        return pixel_result