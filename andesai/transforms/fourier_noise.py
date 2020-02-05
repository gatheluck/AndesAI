import os
import sys
import random

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import torch
import torchvision

class FourierNoise(object):
    def __init__(self, index_h:int, index_w:int, eps:float):
        """
        Args
            index_h: index of fourier basis about hight direction 
            index_w: index of fourier basis about width direction 
            eps: size of noise(perturbation)
        """
        assert abs(index_h)>=1 and abs(index_w)>=1
        assert eps>0.0

        self.index_h = index_h
        self.index_w = index_w
        self.eps = eps

    def __call__(self, x:torch.tensor):
        c,h,w = x.shape[-3:]
        if h%2!=0 or w%2!=0:
            raise ValueError('shape of x is invalid')
        assert -h/2<=self.index_h<=h/2
        assert -w/2<=self.index_w<=w/2

        w_real = torch.zeros_like(x[0,:,:])
        w_imaginary = torch.zeros_like(w_real)

        # make low frequency-centered Fourier palne
        # default Fourier palne is high frequency-centered
        # so swap 1st and 3rd quadrant and swap 2nd and 4th quadrant        
        w_real[self.index_h-1, self.index_w-1] = 1.0
        w_real[h-self.index_h, w-self.index_w] = 1.0

        w = torch.stack([w_real, w_imaginary], dim=-1)

        u = torch.ifft(w, signal_ndim=2)
        u_real = u[:,:,0]
        u_real = u_real/u_real.norm() # normalize

        x[0,:,:] += random.uniform(-1.0,1.0) * self.eps * u_real
        x[1,:,:] += random.uniform(-1.0,1.0) * self.eps * u_real
        x[2,:,:] += random.uniform(-1.0,1.0) * self.eps * u_real

        x = torch.clamp(x, min=0.0, max=1.0)

        return x

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    from andesai.data import DatasetBuilder

    optional_transform = [FourierNoise(2,2,4.0)]

    dataset_builder = DatasetBuilder(name='imagenet100', root_path='../../data')
    test_dataset   = dataset_builder(train=False, normalize=False, optional_transform=optional_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    
    for x,t in test_loader:
        torchvision.utils.save_image(x, '../../logs/fourier_noise_test.png')
        raise NotImplementedError