import os
import sys
import random

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import torch
import torchvision

class PatchSuffle(object):
    def __init__(self, num_divide:int):
        assert num_divide >= 0
        self.num_divide = num_divide

    def __call__(self, x:torch.tensor):
        """
        """
        h = x.size(-2)
        w = x.size(-1)
        dh, dw = h//self.num_divide, w//self.num_divide
        if h%self.num_divide!=0 or w%self.num_divide!=0:
            raise ValueError

        patches = []

        # divide images into patches
        i_pixel_h_start = 0
        for i_patch_h in range(h // dh):
            i_pixel_h_end = i_pixel_h_start+dh

            i_pixel_w_start = 0
            for i_patch_w in range(w // dw):
                i_pixel_w_end = i_pixel_w_start+dw

                # append patche
                patches.append(x[:,i_pixel_h_start:i_pixel_h_end, i_pixel_w_start:i_pixel_w_end])

                i_pixel_w_start = i_pixel_w_end
            i_pixel_h_start = i_pixel_h_end

        random.shuffle(patches)

        # concatenate shuffled patches
        i_start = 0
        imgs = []
        for i in range(self.num_divide):
            i_end = i_start + self.num_divide
            imgs.append(torch.cat(patches[i_start:i_end], dim=-2))
            i_start = i_end 
        img = torch.cat(imgs, dim=-1)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

if __name__ == "__main__":
    from andesai.data import DatasetBuilder

    optional_transform = [PatchSuffle(4)]

    dataset_builder = DatasetBuilder(name='cifar10', root_path='../../data')
    test_dataset   = dataset_builder(train=False, normalize=False, optional_transform=optional_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    
    for x,t in test_loader:
        print(x.shape)
        torchvision.utils.save_image(x, '../../logs/patch_shffle_test.png')
        raise NotImplementedError