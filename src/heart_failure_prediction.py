import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from src.Models.HeartfailureModel import HeartFailureClassifier
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

def train_model(model, save_path, x_train, y_train, x_test, y_test, device, epochs, lr=0.001):
    """
    trains the model on the heartfailure predictions dataset
    :param model: NN for predictions
    :param save_path: saving path
    :param x_train: train data
    :param y_train: train target
    :param x_test: test data
    :param y_test: test target

    """

    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)

    train_data = HeartFailureDataset(x_train,y_train)
    test_data = HeartFailureDataset(x_test,y_test)
    
    train_loader = DataLoader(train_data, batch_size=50, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=False)
    lossF = torch.nn.BCELoss()

    def train(epoch):
        model.train()
        model.to(device)
        train_loss = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = lossF(output, target)
            train_loss += loss
            loss.backward()
            optimizer.step()
        train_loss/=len(train_loader)
        if epoch % 10 == 0:
            print(f"Trainloss: {train_loss}")

    def test():
        model.eval()
        model.to(device)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                test_loss += lossF(output,target)
                pred = (output >= 0.5).int()
                correct += torch.sum(pred.eq(target.int())).item()
        test_loss /= len(test_loader)
        print(f"TestLoss: {test_loss}; Correct: {correct}/{len(test_loader.dataset)}")

    for e in range(epochs):
        test()
        train(e)        
    test() 
    torch.save(model.state_dict(), save_path)


def main():
    datapath = "./data/heart.csv"
    x,y = load_data(datapath)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=42,shuffle=False)
    model = HeartFailureClassifier()
    epochs=40
    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_model(model, save_path="./results/models/heart_failure_1.pth", x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,epochs=epochs, device=device)
    


if __name__ == "__main__":
    main()
