import argparse
from collections import OrderedDict
from torch import nn
from glob import glob
import os


def update_first_layer(model, in_channels, pretrained):
    '''Change first layer based on number of input channels.

    Adapted from fastai/fastai/vision/learner.py'''

    def _get_first_layer(m):
        "Access first layer of a model"
        c,p,n = m,None,None  # child, parent, name
        for n in next(m.named_parameters())[0].split('.')[:-1]:
            p,c=c,getattr(c,n)
        return c,p,n

    def _load_pretrained_weights(new_layer, previous_layer):
        "Load pretrained weights based on number of input channels"
        n_in = getattr(new_layer, 'in_channels')
        if n_in==1:
            # we take the sum
            new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
        elif n_in==2:
            # we take first 2 channels + 50%
            new_layer.weight.data = previous_layer.weight.data[:,:2] * 1.5
        else:
            # keep 3 channels weights and set others to null
            new_layer.weight.data[:,:3] = previous_layer.weight.data
            new_layer.weight.data[:,3:].zero_()

    if in_channels == 3: return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = in_channels
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)


def last_checkpoint(log_dir: str):
    assert os.path.isdir(log_dir)

    checkpoints = sorted(glob(os.path.join(log_dir, "**/*.ckpt")))
    assert len(checkpoints) > 0, f"no checkpoints found in {log_dir}"

    return checkpoints[-1]


def path(path: str):
        if os.path.exists(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid path")


'''Adapted from 
https://discuss.pytorch.org/t/how-do-i-remove-forward-hooks-on-a-module-without-the-hook-handles/140393/2
'''
def remove_all_forward_hooks(model: nn.Module):
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_forward_hooks(child)
