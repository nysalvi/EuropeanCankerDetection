from NeuralNetwork.util.optimizer_factory import Optimizer
from torch.autograd.gradcheck import gradcheck    
import torchvision
import unittest
import logging
import torch
import sys 

class TestNetwork(unittest.TestCase):
    def test__init__(self):                
        self.assertRaises(NotImplementedError, Optimizer)

    def test_GET_OPTIMIZER(self):
                
        x = torch.randn(128, 20,dtype=torch.double, requires_grad=True)
        linear = torch.nn.Linear(20, 1, dtype=torch.double)        
        optimizer = torch.optim.SGD(linear.parameters(), 0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        y = torch.randint(2, (128, 1)).float()

        output = linear(x)
        loss = criterion(output, y)        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
                
        self.assertEqual(True, gradcheck(linear, x))

    def test_grad(self):
                
        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randint(2, (128, 1)).float()
        linear = torch.nn.Linear(20, 1)        
        optimizer = torch.optim.SGD(linear.parameters(), 0.001)
        criterion = torch.nn.BCELoss()
        output = linear(x.float())
        print(output.shape)
        output = torch.nn.Sigmoid()(output)


        manual_loss = []
        for x, r in zip(output, y):
            if r == 1: 
                loss = r - x
            elif r == 0 : 
                loss = x
            manual_loss.append(loss)        
        print(y.shape)
        print(output.shape)
        loss = criterion(output, y.float())        
        manual_loss = torch.tensor(manual_loss)
        print(manual_loss.mean())        
        print(loss)
        self.assertEqual(manual_loss.mean(), loss)
if __name__ == '__main__':
    unittest.main()

