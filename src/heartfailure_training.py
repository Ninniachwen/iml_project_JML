import pandas as pd
import torch
import os
import numpy as np
from pathlib import Path

from typing import Tuple
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from src.classifier.HeartfailureClassifier import HeartFailureClassifier
from src.datasets.heartfailure_dataset import HeartFailureDataset     

# pre processing steps from https://www.kaggle.com/code/yousefahmedsafyelidn/heart-failure-prediction-ann
def load_data(path: Path)->Tuple[torch.Tensor,torch.Tensor]:
    """loads and preprocesses trainign data

    Args:
        path (Path): path to hearts.csv

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: preprocessed x and y values
    """
    df = pd.read_csv(path)
    x = df.drop(columns=['HeartDisease'])
    y = df[['HeartDisease']]
    encoder = LabelEncoder()
    x['ChestPainType'] = encoder.fit_transform(x['ChestPainType'])
    x['Sex'] = encoder.fit_transform(x['Sex'])
    x['ExerciseAngina'] = encoder.fit_transform(x['ExerciseAngina'])
    x['RestingECG'] = encoder.fit_transform(x['RestingECG'])
    x['ST_Slope'] = encoder.fit_transform(x['ST_Slope'])
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    return x,y.values

def train_heartfailure_model(save_path: Path, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, random_seed:int =None, cv:int =0, epochs:int=60, lr:float=0.001,random_dataloader: bool = True):
    """Heartfailure model training method

    Args:
        save_path (Path): Path to folder for saving
        x_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        x_test (np.ndarray): test data
        y_test (np.ndarray): test labels
        random_seed (int, optional): random seed for repoducibility. Defaults to None.
        cv (int, optional): cross-validation parameter. Defaults to 0.
        epochs (int, optional): number of training epochs. Defaults to 60.
        lr (float, optional): learning rate. Defaults to 0.001.
    """
    model = HeartFailureClassifier()

    if random_seed:
        torch.manual_seed(seed=random_seed + cv)

    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)

    train_data = HeartFailureDataset(x_train,y_train)
    test_data = HeartFailureDataset(x_test,y_test)
    
    train_loader = DataLoader(train_data, batch_size=50, shuffle=random_dataloader)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=random_dataloader)
    lossF = torch.nn.BCELoss()

    def train(epoch):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            target = target.to(torch.float32)
            loss = lossF(output, target)
            train_loss += loss
            loss.backward()
            optimizer.step()
        train_loss/=len(train_loader)
        if epoch % 10 == 0:
            print(f"Trainloss: {train_loss}")

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                target = target.to(torch.float32)
                test_loss += lossF(output,target)
                pred = (output >= 0.5).int()
                correct += torch.sum(pred.eq(target.int())).item()
        test_loss /= len(test_loader)
        print(f"TestLoss: {test_loss}; Correct: {correct}/{len(test_loader.dataset)}")

    for e in range(epochs):
        test()
        train(e)        
    test() 
    torch.save(model.state_dict(), os.path.join(save_path,"models", f"model_heartfailure_{cv}.pth"))

