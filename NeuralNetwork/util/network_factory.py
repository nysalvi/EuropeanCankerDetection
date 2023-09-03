from __future__ import nested_scopes
from typing import Iterator, Literal, Type, Tuple, Callable
import torchvision, torch, hydra, re, operator
from hydra.utils import instantiate, call
from ..domain.device import Device
from omegaconf import DictConfig
from torchvision import models
from enum import Enum

class PRE_TRAINED(Enum):
    """        
    Description
    -----------------------------------------------
        Tuple containing network and classification layer
    """
    RESNET = (models.resnet18, models.ResNet18_Weights.DEFAULT),
    RESNET50 = (models.resnet50, models.ResNet18_Weights.DEFAULT),
    VGG16 = (models.vgg16_bn, models.VGG16_BN_Weights.DEFAULT),
    VGG19 = (models.vgg19_bn, models.VGG19_BN_Weights.DEFAULT),
    INCEPTIONV3 = (models.inception_v3, models.Inception_V3_Weights.DEFAULT)

class Network():
    
    def __init__(self):
        """        
        Description
        -----------------------------------------------
            Don't call constructor, use GET_MODEL instead.
            Throws `NotImplementedError` exception
        """
        raise NotImplementedError('This is a static class. Get an instance with Network.GET_MODEL')
        
    def GET_MODEL(network_enum:PRE_TRAINED) -> torch.nn.Module:
        """
        Parameters
        -----------------------------------------------
            network_enum:Models -> One of the PRE_TRAINED enums, available in ./ImageNet.py
            device:str -> Accepted values for device are [auto, cpu, cuda, cuda:{int}, xla]. Auto mode picks the first available option in the reverse order. 
        Description
        -----------------------------------------------
            Loads one of the supported networks from `torchvision.models` and automatically loads the architecture 
            on the given device.
        """           
        model_func, weights = network_enum.value[0]        
        model = model_func(weights=weights)      
        
        return model