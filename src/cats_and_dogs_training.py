import os
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import torch

from src.utils.utlis import CAD_TESTDIR,CAD_TRAINDIR
from src.classifier.CatsAndDogsClassifier import CatsandDogsClassifier
from src.classifier.CatsAndDogsClassifier import CatsandDogsClassifier
from src.classifier.CatsAndDogsClassifier import CatsandDogsClassifier
from src.utils.image_finder_cats_and_dogs import get_images
from src.datasets.cats_and_dogs_dataset import CandDDataSet, augment_image, transform_validate


def train_model(
        save_path: Path,
        cv: int,
        n_epoch: int = 40,
        batch_size_train: int = 128,
        batch_size_test: int = 128,
        batch_size_val: int = 128,
        random_seed: int = 42,
        learning_rate: float = 0.001,
        val_split: float = 0.1,
        data_loader_shuffle = True
)-> CatsandDogsClassifier:
    """CatsandDogsClassifier training methods

    Args:
        save_path (Path): save path for model
        cv (int): cross validation parameter, appends to the save path
        n_epoch (int, optional): amount of training epochs. Defaults to 40.
        batch_size_train (int, optional): batch size for the trainign set. Defaults to 128.
        batch_size_test (int, optional): batch_size for the test set. Defaults to 128.
        batch_size_val (int, optional): batch_size for the validation set. Defaults to 128.
        random_seed (int, optional): random_seed for reproducibility. Defaults to 42.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 0.001.
        val_split (float, optional): relative size of the validation set. Defaults to 0.1.


    Returns:
        CatsandDogsClassifier: saved and trained CatsandDogsClassifier
    """
    # seeded and cudnn set to disabled for reproducibility 
    if random_seed:
        torch.random.manual_seed(random_seed+cv)
        torch.backends.cudnn.enabled = False
    if (val_split > 0.5) or (val_split < 0.05): val_split=0.1
    #path to train and test directory 
    train_dir = CAD_TRAINDIR
    test_dir = CAD_TESTDIR

    picture_files, labels = get_images(train_dir)

    picture_files = [(picture, label) for picture,label in zip(picture_files,labels)]

    val_split = int(val_split*len(picture_files))
    #perform a validation split
    train_files, val_files = random_split(picture_files, [len(picture_files) - val_split, val_split])
    #extraxt data and label fort rain and testset
    val_y = [label for _, label in val_files]
    val_x = [picture_file for picture_file, _ in val_files]
    train_y = [label for _, label in train_files]
    train_x = [picture_file for picture_file, _ in train_files]
    #initialize train and validation set and laoders
    val_set = CandDDataSet(image_paths = val_x, labels = val_y, transform=transform_validate)
    train_set = CandDDataSet(image_paths = train_x, labels = train_y, transform=augment_image)

    train_loader = DataLoader(train_set, batch_size_train, shuffle=data_loader_shuffle)
    val_loader = DataLoader(val_set, batch_size_val, shuffle=data_loader_shuffle)

    picture_files, labels = get_images(test_dir)
    #initialize test set and laoder
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)
    
    test_loader = DataLoader(test_set, batch_size_test, shuffle=data_loader_shuffle)
    
    model = CatsandDogsClassifier()

    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    # binary cross entropy for two class classification problem
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
        print(f"Train epoch {epoch}, Training loss = {train_losses[-1]:.4f}")

    def validate():
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data,target in val_loader:

                target = target.to(torch.float32)

                output = model(data)
                output = output.squeeze()
                
                val_loss += lossF(output,target).item()
                
                pred = (output>=0.5).int()
                correct += torch.sum(pred.eq(target.int())).item()

        val_loss /= len(val_loader)
        val_accs.append(correct/len(val_loader.dataset))
        val_losses.append(val_loss)
        print(f"Avgerage validation loss. {val_losses[-1]:.4f}, Validation accuracy: {correct}/{len(val_loader.dataset)} = {val_accs[-1]:.4f}")

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
        print(f"Average test loss. {test_losses[-1]:.4f}, Testaccuracy: {correct}/{len(test_loader.dataset)} = {test_accs[-1]:.4f}")
        

    test()
    epoch=0
    for epoch in range(1,n_epoch+1):
        train(epoch)
        validate()
        scheduler.step(val_losses[-1])
    test() 
    # save model 
    torch.save(model.state_dict(), os.path.join(save_path,"models", f"model_cad_{cv}.pth"))
    return model

