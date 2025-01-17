import torch
import torch.nn as nn

from original_code.src.simplexai.models.base import BlackBox

class CatsandDogsClassifier(BlackBox):
    def __init__(self)-> None:
        """
        pets and dogs CNN classifier. Uses three convolutional layers and two linear layer for flattening.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3)
        self.relu= nn.ReLU()
        self.conv3 = nn.Conv2d(32,64,3)
        self.fc1 = nn.Linear(64*17*17,128)
        self.fc2 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()
        

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes latent represenation
        :param x: input tensor
        :return: latent representation as tensor
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return x

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        forward pass
        :param x: input tensor
        :return: model output
        """
        x = self.latent_representation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def latent_to_presoftmax(self, h: torch.Tensor) -> torch.Tensor:
        """
        Maps a latent representation to a preactivation output
        :param h: latent representations
        :return: presoftmax activations
        """
        return self.fc2(h)
    
    




    




