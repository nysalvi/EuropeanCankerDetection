import torchvision
import torch

def resnet(self:torchvision.models.ResNet):
    in_features = self.fc.in_features
    self.fc = torch.nn.Linear(in_features=in_features, out_features=1)

def vgg(self:torchvision.models.VGG):
    in_features = self.classifier[6].in_features
    self.classifier[6] = torch.nn.Linear(in_features=in_features, out_features=1)

def inceptionv3(self:torchvision.models.Inception3):
    in_features = self.fc.in_features
    self.fc = torch.nn.Linear(in_features=in_features, out_features=1)

torchvision.models.ResNet.set_classifier = resnet
torchvision.models.VGG.set_classifier = vgg
torchvision.models.Inception3.set_classifier = inceptionv3

                                    