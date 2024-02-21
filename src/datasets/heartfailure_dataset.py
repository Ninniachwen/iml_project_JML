import torch
from torch.utils.data import Dataset

# source: TODO
class HeartFailureDataset(Dataset):
    """
    HeartFailure Dataset
    """
    def __init__(self, x, y) -> None:
        self.X = x
        self.y = y.astype(int)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.X[idx],dtype=torch.float32)
        target = self.y[idx]
        return data, target