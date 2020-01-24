import os
import sys
import math

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import torch
import torch.nn as nn
import torchvision

from andesai.data import DatasetBuilder
DATASET_CONFIG = DatasetBuilder.DATASET_CONFIG

def get_step_size(epsilon, n_iters, use_max=False):
    if use_max:
        return epsilon
    else:
        return epsilon / math.sqrt(n_iters)

def get_eps_params(base_eps, resol):
    eps_list = []
    max_list = []
    min_list = []
    for i in range(3):
        eps_list.append(torch.full((resol, resol), base_eps, device='cuda'))
        min_list.append(torch.full((resol, resol), 0., device='cuda'))
        max_list.append(torch.full((resol, resol), 255., device='cuda'))

    eps_t = torch.unsqueeze(torch.stack(eps_list), 0)
    max_t = torch.unsqueeze(torch.stack(max_list), 0)
    min_t = torch.unsqueeze(torch.stack(min_list), 0)
    return eps_t, max_t, min_t

def get_dataset_params(dataset):
    resol = DATASET_CONFIG[dataset].input_size
    dataset_mean = DATASET_CONFIG[dataset].mean
    dataset_std  = DATASET_CONFIG[dataset].std

    mean_list = []
    std_list = []
    for i in range(3):
        mean_list.append(torch.full((resol, resol), dataset_mean[i], device='cuda'))
        std_list.append(torch.full((resol, resol), dataset_std[i], device='cuda'))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)

class DatasetTransform(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.mean, self.std = get_dataset_params(dataset)
        
    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to [0, 255]
        '''
        x = x / 255.
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class InverseDatasetTransform(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.mean, self.std = get_dataset_params(dataset)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - MEAN) / STD
        '''
        x = x.mul(self.std)
        x = x.add(self.mean)
        x = x * 255.
        return x

class PixelModel(nn.Module):
    def __init__(self, model, dataset):
        super().__init__()
        if dataset not in DATASET_CONFIG.keys():
            raise ValueError('dataset name is invalid')

        self.model = model
        self.transform = DatasetTransform(dataset)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to [0, 255]
        '''
        x = self.transform(x)
        # x is now normalized as the model expects
        x = self.model(x)
        return x

class AttackWrapper(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        if dataset not in DATASET_CONFIG.keys():
            raise ValueError('dataset name is invalid')

        self.dataset = dataset
        self.resol = DATASET_CONFIG[dataset].input_size
        self.transform = DatasetTransform(dataset)
        self.inverse_transform = InverseDatasetTransform(dataset)
        self.epoch = 0
        
    def forward(self, model, img, *args, **kwargs):
        was_training = model.training
        pixel_model = PixelModel(model, self.dataset)
        pixel_model.eval()
        pixel_img = self.inverse_transform(img.detach())
        pixel_ret = self._forward(pixel_model, pixel_img, *args, **kwargs)
        if was_training:
            pixel_model.train()
        ret = self.transform(pixel_ret)
        return ret

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._update_params(epoch)

    def _update_params(self, epoch):
        pass