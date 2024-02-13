import torch
from torch.utils.data import Dataset

class HeartFailureDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.X = x
        self.y = y.astype(int)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.X[idx],dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return data, target