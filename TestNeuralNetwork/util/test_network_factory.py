from NeuralNetwork.util.network_factory import Network, PRE_TRAINED
import torchvision
import unittest
import torch
import logging
import sys 
class TestNetwork(unittest.TestCase):
    def test__init__(self):                
        self.assertRaises(NotImplementedError, Network)

    def test_GET_MODEL(self):

        resnet18 = torchvision.models.resnet18(pretrained=True)
        num_features = resnet18.fc.in_features 
        resnet18.fc = torch.nn.Linear(num_features, 1) 

        test_model = Network.GET_MODEL(PRE_TRAINED.RESNET)
        for x, y in zip(resnet18.parameters(), test_model.parameters()):
            self.assertEqual(x.data.all(), y.data.all())

if __name__ == '__main__':
    unittest.main()
