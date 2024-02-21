import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import os
from torch.utils.data import DataLoader

from src.classifier.HeartfailureClassifier import HeartFailureClassifier
from src.datasets.heartfailure_dataset import HeartFailureDataset     


def load_data(path):
    """
    Loads and preprocesses data, encodes categorical data and performs minmax scaling
    :param path: path to heartdisease csv file
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

def train_heartfailure_model(model, save_path, x_train, y_train, x_test, y_test, random_seed=None, cv=0, epochs=60, lr=0.001):
    """
    trains the model on the heartfailure predictions dataset
    :param model: NN for predictions
    :param save_path: saving path
    :param x_train: train data
    :param y_train: train target
    :param x_test: test data
    :param y_test: test target

    """
    if random_seed:
        torch.manual_seed(seed=random_seed + cv)

    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)

    train_data = HeartFailureDataset(x_train,y_train)
    test_data = HeartFailureDataset(x_test,y_test)
    
    train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=True)
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
    torch.save(model.state_dict(), os.path.join(save_path, f"model_heartfailure_{cv}.pth"))

