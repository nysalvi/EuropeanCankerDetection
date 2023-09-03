from NeuralNetwork.util.network_factory import Network, PRE_TRAINED
from NeuralNetwork.util.optimizer_factory import Optimizer
from NeuralNetwork.domain.device import Device
from NeuralNetwork.domain.train import Train
from NeuralNetwork.service.filer import File
from NeuralNetwork.service.result import Result
from hydra.utils import instantiate, call
from torchvision import transforms
from omegaconf import DictConfig
from tqdm.auto import tqdm
from typing import Union
import pandas as pd
import torchvision
import operator
import re, os
import hydra
import torch
import sys

def load_datasets(path, cfgDataAugment:Union[dict, DictConfig], batch_size):
    train_augment = eval(cfgDataAugment['train'])        
    dev_augment = eval(cfgDataAugment['dev'])        
    test_augment = eval(cfgDataAugment['test'])        
    
    trainset = File.loadDataset(os.path.join(path, 'train'), train_augment, batch_size)
    devset = File.loadDataset(os.path.join(path, 'dev'), dev_augment, batch_size)
    testset = File.loadDataset(os.path.join(path, 'test'), test_augment, batch_size)

    return trainset, devset, testset    

def get_model(name:str) -> torch.nn.Module:
    model_name = PRE_TRAINED[name]        
    model = Network.GET_MODEL(model_name)
    model.set_classifier()    
    return Device.TO_DEVICE(model)

def start_config(cfg:DictConfig):
    filer = File(cfg.path)
    if cfg.restart: filer.clear_progress()
    checkpoint = filer.load_checkpoint()
    if 'state' in checkpoint : 
        state = checkpoint['state'] 
        if state['finished']: exit('Already finished training')
    else : 
        state = {
            'epoch':0, 'tol': 0, 'finished':0, 'lr':cfg.lr, 'path':cfg.path,
            'device':cfg.device, 'max_tol':cfg.total_tol, 'metric' : cfg.metric,
            'max_epoch':cfg.total_epochs, 'weight_decay':cfg.wd, 'value':0
        }
    return checkpoint, filer, state

def update_config(state, dev_data, save_metric):    
    state['epoch']+= 1
    value = dev_data['save_metric'].round(4)
    if state['value'] < value:
        state['value'] = value
        state['tol'] = 0 
    else: 
        state['tol']+=1

    if state['epoch'] == state['max_epoch'] or state['tol'] == state['max_tol']:
        state['finished'] = 1
def get_network_state(training:Train):
    model_state = training.model.state_dict()
    optim_state = training.optimizer.state_dict()
    scheduler_state = training.scheduler.state_dict()    
    return {'model':model_state, 'optim':optim_state, 'lr_scheduler':scheduler_state}


@hydra.main(version_base='1.3.2', config_path='config', config_name='config.yaml')
def main(cfg:DictConfig):   
    checkpoint, filer, state = start_config(cfg)

    model = get_model(cfg.model.name)            
    if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
    optimizer = Optimizer.GET_OPTIMIZER(model, cfg.optimizer)
    if 'optim' in checkpoint: optimizer.load_state_dict(checkpoint['optim'])    
    lr_scheduler = call(cfg.lr_scheduler, optimizer=optimizer, last_epoch=state['epoch'] - 1)    
    if 'lr_scheduler' in checkpoint: lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 
    criterion = call(cfg.criterion)
    training = Train(model, optimizer, lr_scheduler, criterion)

    dataset_path = os.path.join(os.getcwd(), 'data')       
    trainset, devset, testset = load_datasets(dataset_path, cfg.data_augment, cfg.batch_size)
    pbar = tqdm(range(state['epoch'], state['max_epoch']))            
    
    for e in pbar:        

        train_resul = training.train(trainset)
        dev_resul = training.eval(devset)
        
        stats_train = Result(*train_resul).create_metrics()        
        stats_dev = Result(*dev_resul).create_metrics()
                
        df = pd.concat(stats_train.data, stats_dev.data) ; df['epoch'] = e

        update_config(state, stats_dev.data, cfg.save_metric)

        if state['tol'] == 0: network_state = get_network_state(training)        
        filer.save_checkpoint(df, {'state': state, **network_state})
        if state['finished']: break

    test_resul = training.eval(testset)        
    stats_test = Result(*test_resul).create_metrics()
    filer.save_checkpoint(stats_test.data)

if __name__ == '__main__':    
    main()