from omegaconf import DictConfig
from torchvision import models
from ..domain.device import Device
from enum import Enum
from typing import Iterator, Type, Tuple, Callable
from hydra.utils import instantiate, call
import torch

class Optimizer():
    def __init__(self):
        """        
        Description
        -----------------------------------------------
            Don't call constructor, use GET_OPTIMIZER instead.
            Throws `NotImplementedError` exception
        """        
        raise NotImplementedError('This is a static class. Get an instance with `Optimizer.GET_OPTIMIZER` ')
    def GET_OPTIMIZER(model:torch.nn.Module, cfgOptim:DictConfig, all_params:bool=True,
            custom_grad:Iterator[torch.nn.parameter.Parameter]=None) -> torch.optim.Optimizer:                    
        if custom_grad : params = custom_grad(model) 
        elif all_params: params = model.parameters()                
        return call(cfgOptim, params)