from pickletools import optimize
from torchvision import models

import torch.optim as optim
import torch.utils.data
import torch

class Train():

    def __init__(self, model:torch.nn.Module, optimizer:optim.Optimizer=optim.SGD, 
                 scheduler=optim.lr_scheduler.CosineAnnealingLR, criterion:torch.nn.Module=torch.nn.BCEWithLogitsLoss()):
        self.model = model        
        self.optimizer = optimizer
        self.scheduler = scheduler        
        self.criterion = criterion
    def train(self, loader:torch.utils.data.DataLoader):
        self.model.train()
        all_logits = torch.zeros()        
        all_y = torch.zeros()

        x:torch.Tensor; y:torch.Tensor 
        output:torch.Tensor 
        
        for i, x, y in enumerate(loader):            
            self.optimizer.zero_grad()
            x, y = x.to(self.model.device), y.to(self.model.device)            
            if self.model._get_name() == "Inception3": 
                output = self.model(x)[0].squeeze()
            else : output = self.model(x).squeeze()             

            loss:torch.Tensor = self.criterion(output, y.float())                         
            loss.backward()            
            self.optimizer.step()        

            all_logits[i*len(x):i*len(x)+len(x)] = output.detach()
            all_y[i*len(y):i*len(y)+len(y)] = y.detach()            
        self.scheduler.step()
        return all_logits, all_y

    def eval(self, loader:torch.utils.data.DataLoader):
        self.model.eval()        

        all_logits = torch.zeros()        
        all_y = torch.zeros()

        with torch.no_grad():
            output:torch.Tensor
            for i, x, y in enumerate(loader):
                x, y = x.to(self.model.device), y.to(self.model.device)
                output = self.model(x).squeeze()             
                all_logits[i*len(x):i*len(x)+len(x)] = output
                all_y[i*len(y):i*len(y)+len(y)] = y

        return all_logits, all_y
    
