import inspect
import os
import sys
import torch
from torch.utils.data import DataLoader

# access model in parent dir: https://stackoverflow.com/a/11158224/14934164
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)

import sys
sys.path.insert(0, "")  # noqa

from Original_Code.src.simplexai.models.image_recognition import MnistClassifier
from Original_Code.src.simplexai.experiments import mnist
from src.cats_and_dogs_training import train_model, load
from src.models.CatsAndDogsModel import CatsAndDogsClassifier
from src.cats_and_dogs_predictions import load_model
from src.datasets.cats_and_dogs_dataset import CandDDataSet
from src.utils.image_finder_cats_and_dogs import get_images
from src.utils.corpus_creator import make_corpus


SAVE_PATH=os.path.join(parentdir, "files")


def train_or_load_mnist(random_seed=42, cv=0, corpus_size=100, test_size=10, random_dataloader=False):
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
    # model.to(device) porbably not necessary
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
    #TODO: maybe implement own mnist latents fuction?
    #TODO: which detach is truely necessary?
    
    return classifier, (corpus_data, corpus_latents, corpus_target), (test_data, test_targets, test_latents)


def train_or_load_new_dataset_model(random_seed=42, cv=0, corpus_size=100, test_size=10, random_dataloader=False):
    
    torch.manual_seed(seed=random_seed)
    
    save_path = os.path.join(SAVE_PATH,f"model_cad{cv}.pth")
    if not os.path.isfile(save_path):
        train_model(save_path=save_path, random_seed=random_seed)
    
    classifier = load_model(save_path)
    
    test_dir = r"data\Animal Images\test"

    picture_files, labels = get_images(test_dir)
    
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)

    test_loader = DataLoader(test_set, batch_size=200, shuffle=True)

    (test_data, test_targets) = make_corpus(test_loader, corpus_size=10)

    test_data = test_data.detach()
    test_latents = classifier.latent_representation(test_data).detach()

    corpus_dir = r"data\Animal Images\train"

    picture_files = []

    picture_files, labels = get_images(corpus_dir)
    
    corpus_set = CandDDataSet(image_paths=picture_files, labels=labels)

    corpus_loader = DataLoader(corpus_set, batch_size=50, shuffle=True)

    (corpus_data, corpus_target) = make_corpus(corpus_loader=corpus_loader)

    corpus_data = corpus_data.detach()
    corpus_latents = classifier.latent_representation(corpus_data).detach()

    return classifier, (corpus_data, corpus_latents, corpus_target), (test_data, test_targets, test_latents)
