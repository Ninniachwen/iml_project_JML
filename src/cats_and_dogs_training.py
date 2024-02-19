import os
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.Models.CatsAndDogsModel import CatsandDogsClassifier
from src.utils.image_finder_cats_and_dogs import get_images
from src.datasets.cats_and_dogs_dataset import CandDDataSet, augment_image, transform_validate, LABEL



def train_model(
        save_path: Path,
        cv: int,
        n_epoch: int = 60,
        batch_size_train: int = 128,
        batch_size_test: int = 128,
        batch_size_val: int = 128,
        random_seed: int = 42,
        learning_rate: float = 0.001,
        val_split: float = 0.1
)-> CatsandDogsClassifier:
    
    # seeded and cudnn set to disabled for reproducibility 
    torch.random.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    
    train_dir = r"data\Animal Images\train"
    test_dir = r"data\Animal Images\test"

    picture_files, labels = get_images(train_dir)

    val_split = int(val_split*len(picture_files))

    train_files, val_files = random_split(picture_files, [len(picture_files) - val_split, val_split])

    val_y = [label for _, label in val_files]
    val_x = [picture_file for picture_file, _ in val_files]
    train_y = [label for _, label in train_files]
    train_x = [picture_file for picture_file, _ in train_files]

    val_set = CandDDataSet(image_paths = val_x, labels = val_y, transform=transform_validate)
    train_set = CandDDataSet(image_paths = train_x, labels = train_y, transform=augment_image)


    picture_files, labels = get_images(test_dir)
    
    
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)

    train_loader = DataLoader(train_set, batch_size_train, shuffle=True)
    val_loader = DataLoader(val_set, batch_size_val, shuffle=False)
    test_loader = DataLoader(test_set, batch_size_test, shuffle=False)

    
    model = CatsandDogsClassifier()

    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate,)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    
    lossF = torch.nn.BCELoss(reduction='mean')
    train_losses = []
    test_losses = []
    val_losses = []
    val_accs = []
    test_accs = []

    def train(epoch):
        model.train()
        train_loss = 0
        for data,target in train_loader:

            optimizer.zero_grad()

            target = target.to(torch.float32)      
            output = model(data)
            output = output.squeeze()
            
            loss = lossF(output,target)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"Train epoch {epoch}, Loss = {train_losses[-1]}")


    def validate():
        model.eval()
        val_loss = 0
        correct = 0
        balance = 0
        with torch.no_grad():
            for data,target in val_loader:

                target = target.to(torch.float32)

                output = model(data)
                output = output.squeeze()
                
                val_loss += lossF(output,target).item()
                
                pred = (output>=0.5).int()
                balance += torch.mean(pred.float()).item()
                correct += torch.sum(pred.eq(target.int())).item()

        balance /= len(val_loader)
        val_loss /= len(val_loader)
        val_accs.append(correct/len(val_loader.dataset))
        val_losses.append(val_loss)
        print(f"Avgerage validation loss. {val_losses[-1]}, Valaccuracy: {correct}/{len(val_loader.dataset)} = {val_accs[-1]} , Balance: {balance}")

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data,target in test_loader:

                target = target.to(torch.float32)

                output = model(data)
                output = output.squeeze()
                
                test_loss += lossF(output,target).item()
                
                pred = (output>=0.5).int()

                correct += torch.sum(pred.eq(target.int())).item()
        test_loss /= len(test_loader)
        test_accs.append(correct/len(test_loader.dataset))
        test_losses.append(test_loss)
        print(f"Average test loss. {test_losses[-1]}, Testaccuracy: {correct}/{len(test_loader.dataset)}")
        

    test()
    epoch=0
    for epoch in range(1,n_epoch+1):
        train(epoch)
        validate()
        scheduler.step(val_losses[-1])
    test() 
    
    torch.save(optimizer.state_dict(), save_path)
    return model

