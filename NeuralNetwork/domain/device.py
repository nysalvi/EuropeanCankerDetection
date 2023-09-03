#import torch_xla.core.xla_model as xm
from enum import Enum
#import torch_xla
import torch
import re

class Device():
    ALLOWED_DEVICE = ['cpu', 'cuda', 'auto']
    GET_DEVICE = [
        torch.device('cpu'),
        torch.device('cuda')
    ]
    def __init__(self):
        """        
        Description
        -----------------------------------------------
            Don't call constructor, use TO_DEVICE instead.
            Throws `NotImplementedError` exception
        """
        raise NotImplementedError()
    def __auto__(model:torch.nn.Module):
        """        
        Description
        -----------------------------------------------
            Automatically picks device, with priority xla > cuda > cpu            
            NOTE: xla currently not implemented!!!
        """                
        for x in range(len(Device.GET_DEVICE) - 1, -1):
            try:
                model.to(Device.GET_DEVICE[x])                
                model.device = Device.GET_DEVICE[x]
                return model
            except: pass
            
    def TO_DEVICE(model:torch.nn.Module, device:str='auto') -> torch.nn.Module:      
        """
        Parameters
        -----------------------------------------------
            model:torch.nn.Module -> pytorch module class 
            device:str -> One of the following values ['cpu', 'gpu', 'xla', 'auto']. 
            
        Description
        ----------------------------------------------- 
            Applies specified device on model. For 'auto', priority is xla > cuda > cpu

            NOTE: xla is currently not supported
        """
        i = 0
        if not any(re.match(pattern, device) for i, pattern in enumerate(Device.ALLOWED_DEVICE)):
            raise ValueError(f'Accepted values for device are {Device.ALLOWED_DEVICE}')
            
        if device == 'auto': return Device.__auto__(model) 
        else: 
            model.to(Device.GET_DEVICE[i])
            model.device = Device.GET_DEVICE[i]
            return model
