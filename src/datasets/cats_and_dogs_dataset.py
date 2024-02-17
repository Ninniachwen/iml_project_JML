import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2

LABEL ={"cats" : 0, "dogs" : 1}

transform_normal = v2.Compose([   
    v2.ToImage(),
    v2.Resize((150, 150)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,],std=[0.5,])
])

transform_validate = v2.Compose([   
    v2.ToImage(),
    v2.CenterCrop(size=(130,130)),
    v2.Resize((150, 150)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,],std=[0.5,])
])

augment_image = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(size=(130,130)),
    v2.RandomHorizontalFlip(),
    v2.Resize(size=(150,150)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,],std=[0.5,])
]) 


class CandDDataSet(Dataset):
    def __init__(self, image_paths, labels, subset_size=None, transform=transform_normal):
        self.image_paths = image_paths
        self.labels = labels
        self.subset_size = subset_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.image_paths[idx]
        label = self.labels[idx] 
        
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label
