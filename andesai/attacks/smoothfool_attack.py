import os
import sys
import random

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import numpy as np
import torch

from collections import OrderedDict

from andesai.attacks.attacks import AttackWrapper


class SmoothFoolAttack(AttackWrapper):
    def __init__(self, dataset:str, nb_its:int, eps_max:float, ):
        """
        Parameters:
            dataset (str):         dataset name.
            nb_its (int):          Number of max SmoothFool iterations.
            eps_max (float):       The max norm, in pixel space.
        """
        super().__init__(dataset)
        self.nb_its = nb_its
        self.eps_max = eps_max
        #self.deepfool_attacker = 

    def _smooth_cliping(self):

    def _normal_cliping(self):

    def _run_one(self):

    def _forward(self, pixel_model, pixel_img, target, avoid_target=True):
        """
        Args
        - pixel_model: model which able to take pixel space image [0,255] as input.
        - pixel_img: image in pixel space [0,255]
        - 
        Returns
        - 
        """
        info_dict = OrderedDict()

        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True
        delta = self._init(pixel_inp.size(), base_eps)

        while it < self.nb_its and 

        pixel_result = 
        return pixel_result, info_dict

def smoothfool(net, im, alpha_fac, dp_lambda, smoothing_func, max_iters=500, smooth_clipping=True, device='cuda'):
    net = net.to(device)
    im = im.to(device)
    x_i = copy.deepcopy(im).to(device)
    loop_i = 0
    f_image = net.forward(Variable(im[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    label_nat = np.argmax(f_image)
    k_i = label_nat
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
    total_clip_iters = 0
    attck_mon = []
    while loop_i < max_iters and k_i == label_nat:
        normal, x_adv, adv_lbl = deepfool(x_i[None, :, :, :], net, lambda_fac=dp_lambda, num_classes=10, device=device)
        normal_smooth = smoothing_func(normal)
        normal_smooth = normal_smooth / torch.norm(normal_smooth.view(-1))
        dot0 = torch.dot(normal.view(-1), x_adv.view(-1) - x_i.view(-1))
        dot1 = torch.dot(normal.view(-1), normal_smooth.view(-1))
        alpha = (dot0 / dot1) * alpha_fac
        normal_smooth = normal_smooth * alpha

        clip_iters = 0
        if smooth_clipping:
            normal_smooth, clip_iters = smooth_clip(x_i[None, :, :, :], normal_smooth, smoothing_func)
            if clip_iters > 198:
                print("clip_iters>iters_max")
                break
            total_clip_iters += clip_iters
            x_i = x_i + normal_smooth[0, :, :, :]
        else:
            x_i = clip_value(x_i + normal_smooth[0, ...])

        f_image = net.forward(Variable(x_i[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        k_i = np.argmax(f_image)
        loop_i += 1
        print("         step: %03d, predicted label: %03d, prob of pred: %.3f, n of clip iters: %03d" % (
            loop_i, k_i, np.max(f_image), clip_iters))
        attck_mon.append(np.max(f_image))

        # track the performance of attack
        if len(attck_mon) > 10:
            del attck_mon[0]

    return x_i, loop_i, total_clip_iters, label_nat, k_i
