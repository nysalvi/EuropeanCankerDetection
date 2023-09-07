from __future__ import annotations
from sklearn.metrics import roc_curve, auc, fbeta_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
import torch

class Result():
    def fbeta(self):
        return fbeta_score(self.y, self.pred, beta=0.5)
    
    def loss(self):
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(self.logit, self.y)
        return loss

    def auc(self):        
        fpr, tpr, _ = roc_curve(self.y, self.score)
        return auc(fpr, tpr)
        
    SKLEARN_METRICS = [        
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    ]
    METRICS = [
        loss,
        fbeta,
        auc
    ]    
    def __init__(self, x_logit:torch.Tensor, y:torch.Tensor):        
        self.logit = np.array(x_logit)
        self.score = np.array(torch.nn.Sigmoid()(x_logit))
        self.pred = np.array((x_logit>=0).int())
        self.y = np.array(y)
        
    def __init__(self, data):        
        self.data = data
        self.logit = np.array(data['logit'])
        self.score = np.array(data['score'])
        self.pred = np.array(data['pred'])
        self.y = np.array(data['y'])

    @staticmethod
    def from_csv(path) -> Result: 
        data = pd.read_csv(path)         
        return Result(data)

    def create_metrics(self):
        temp = {}
        for func in Result.SK_METRICS:
            temp.update({func.__name__.removesuffix('_score') : func(self.y, self.pred)})
        for func in Result.METRICS:
            temp.update({func.__name__: func()})
        self.data = pd.DataFrame.from_dict(temp)        
        return self

    def save(self, path):        
        self.data.to_csv(path)