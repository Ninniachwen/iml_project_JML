import torch
import torch.nn as nn
import torch.nn.functional as F

from original_code.src.simplexai.models.base import BlackBox

class HeartFailureClassifier(BlackBox):
    def __init__(self)-> None:
        """
        Heart Failure Prediction Neural net
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features=11,out_features=16)
        self.fc2 = nn.Linear(in_features=16,out_features=8)
        self.fc3 = nn.Linear(in_features=8,out_features=8)
        self.fc4 = nn.Linear(in_features=8,out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes latent represenation
        :param x: input tensor
        :return: latent representation as tensor
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        forward pass
        :param x: input tensor
        :return: model output
        """
        x = self.latent_representation(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x