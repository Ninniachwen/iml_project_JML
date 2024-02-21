import inspect
import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys

from src.cats_and_dogs_predictions import load_model

# access model in parent dir: https://stackoverflow.com/a/11158224/14934164
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

sys.path.insert(0, "")

from original_code.src.simplexai.models.image_recognition import MnistClassifier
from original_code.src.simplexai.experiments import mnist
from src.cats_and_dogs_training import train_model
from src.classifier.CatsAndDogsClassifier import CatsandDogsClassifier
from src.datasets.cats_and_dogs_dataset import CandDDataSet
from src.utils.image_finder_cats_and_dogs import get_images
from src.utils.corpus_creator import make_corpus
from src.heartfailure_prediction import train_heartfailure_model, load_data
from src.datasets.heartfailure_dataset import HeartFailureDataset
from src.classifier.HeartfailureClassifier import HeartFailureClassifier


SAVE_PATH=os.path.join(parentdir, "files")


def train_or_load_mnist(random_seed=42, cv=0, corpus_size=100, test_size=10, random_dataloader=False, use_corpus_maker=False) -> tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Loads the mnist classifier if one is stored or trains one instead. Also loads the mnist dataset

    Args:
        random_seed (int, optional): random seed for seeding the mnist model. Defaults to 42.
        cv (int, optional): is added to random seed 42, to run experiments with different random seeds. used as identifier for stored models. Defaults to 0.
        corpus_size (int, optional): How many images to use for corpus (explainer images). Defaults to 100.
        test_size (int, optional): How many images to use for test_set (images that will be explained, using corpus). Defaults to 10.
        random_dataloader (bool, optional): Whether samples random images or the same images in each run. Defaults to False.
        use_corpus_maker (bool, optional): Whether to use the corpus maker (randomizing data). Defaults to False.#TODO: lucas, reicht die erklärung?

    Returns:
        tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents): Data is a tensor representation of one or more images. Target is an integer representing the images class. Latents are latent representation from the mnist classifier.
    """
    # the following are standard values which are used in mnist.py to train mnistClassifier
    if not os.path.isfile(os.path.join(SAVE_PATH,f"model_cv{cv}.pth")):
        mnist.train_model(
                device="cpu",
                random_seed=random_seed,
                cv=cv,
                save_path=SAVE_PATH,
                model_reg_factor=0.1,
            )
        
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(SAVE_PATH,f"model_cv{cv}.pth")))
    classifier.eval()

    # data loader from approximate_quality
    corpus_loader = mnist.load_mnist(corpus_size, train=True, shuffle=random_dataloader)
    test_loader = mnist.load_mnist(test_size, train=False, shuffle=random_dataloader)
    batch_id_test , (test_data, test_targets) = next(enumerate(test_loader))
    test_data = test_data.detach()
    test_latents = classifier.latent_representation(test_data).detach()
    batch_id_corpus, (corpus_data, corpus_target) = next(enumerate(corpus_loader))
    corpus_data = corpus_data.detach()
    corpus_latents = classifier.latent_representation(corpus_data).detach()

    #TODO: use corpus maker (Lucas)

    return classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents)


def train_or_load_CaD_model(random_seed=42, cv=0, corpus_size=100, test_size=10, random_dataloader=False):

    torch.manual_seed(seed=random_seed)
    
    if not os.path.isfile(os.path.join(SAVE_PATH,f"model_cad_{cv}.pth")):
        train_model(save_path=SAVE_PATH, cv=cv, random_seed=random_seed)

    classifier = load_model(os.path.join(SAVE_PATH,f"model_cad_{cv}.pth"))
    classifier.eval()
    
    test_dir = r"data\Animal Images\test"

    picture_files, labels = get_images(test_dir)
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)
    test_loader = DataLoader(test_set, batch_size=200, shuffle=True)
    (test_data, test_targets) = make_corpus(test_loader, corpus_size=test_size)
    test_data = test_data.detach()
    test_latents = classifier.latent_representation(test_data).detach()

    corpus_dir = r"data\Animal Images\train"

    picture_files, labels = get_images(corpus_dir)
    corpus_set = CandDDataSet(image_paths=picture_files, labels=labels)
    corpus_loader = DataLoader(corpus_set, batch_size=200, shuffle=True)
    (corpus_data, corpus_target) = make_corpus(corpus_loader=corpus_loader, corpus_size=corpus_size)
    corpus_data = corpus_data.detach()
    corpus_latents = classifier.latent_representation(corpus_data).detach()

    return classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents)


def train_or_load_heartfailure_model(random_seed=42, cv=0, corpus_size=100, test_size=10, random_dataloader=False):
    torch.manual_seed(random_seed)

    datapath = r"data\heart.csv"

    x,y = load_data(datapath)
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1, random_state=42, shuffle=random_dataloader)

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1, random_state=42, shuffle=random_dataloader)
    classifier = HeartFailureClassifier()
    file_w_path = os.path.join(SAVE_PATH, f"model_heartfailure_{cv}.pth")
    if not os.path.isfile(file_w_path):
        train_heartfailure_model(classifier, save_path=file_w_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, cv=cv)

    classifier.load_state_dict(torch.load(file_w_path))
    classifier.eval()

    train_data = HeartFailureDataset(x_train, y_train)
    test_data = HeartFailureDataset(x_test, y_test)
    
    train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=True)

    (corpus_data, corpus_target) = make_corpus(train_loader, corpus_size=corpus_size)

    (test_data, test_targets) = make_corpus(test_loader, corpus_size=test_size)

    corpus_data = corpus_data.detach()
    test_data = test_data.detach()
    corpus_latents = classifier.latent_representation(corpus_data).detach()
    test_latents = classifier.latent_representation(test_data).detach()

    return classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents)


if __name__ == "__main__":
    train_or_load_heartfailure_model()
    train_or_load_CaD_model()