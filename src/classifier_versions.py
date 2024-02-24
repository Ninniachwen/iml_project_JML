import inspect
import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, "")
from original_code.src.simplexai.models.image_recognition import MnistClassifier
from original_code.src.simplexai.experiments import mnist
from src.cats_and_dogs_training import train_model, load_model
from src.utils.utlis import CAD_TESTDIR,CAD_TRAINDIR, HEART_FAILURE_DIR
from src.datasets.cats_and_dogs_dataset import CandDDataSet
from src.utils.image_finder_cats_and_dogs import get_images
from src.utils.corpus_creator import make_corpus
from src.heartfailure_training import train_heartfailure_model, load_data
from src.datasets.heartfailure_dataset import HeartFailureDataset
from src.classifier.HeartfailureClassifier import HeartFailureClassifier

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
SAVE_PATH=os.path.join(parentdir, "files")


def train_or_load_mnist(random_seed:int = 42, cv: int = 0, corpus_size:int=100, test_size:int=10, random_dataloader: bool=False, use_corpus_maker=False) -> tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
    if use_corpus_maker:
        corpus_data, corpus_target = make_corpus(corpus_loader, corpus_size=corpus_size, n_classes=10)
        test_data, test_targets = make_corpus(test_loader, corpus_size=corpus_size, n_classes=10)
    else:
        batch_id_corpus, (corpus_data, corpus_target) = next(enumerate(corpus_loader))
        corpus_data = corpus_data.detach()
        batch_id_test, (test_data, test_targets) = next(enumerate(test_loader))
        test_data = test_data.detach()
        
    test_latents = classifier.latent_representation(test_data).detach()
    corpus_latents = classifier.latent_representation(corpus_data).detach()

    return classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents)


def train_or_load_CaD_model(random_seed: int=42, cv: int =0, corpus_size: int=100, test_size: int=10, random_dataloader: bool=False) -> tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """trains or loads the CatsandDogsClassifier

    Args:
        random_seed (int, optional): random seed for reproducibility. Defaults to 42.
        cv (int, optional): cross validation parameter. Defaults to 0.
        corpus_size (int, optional): size of the corpus. Defaults to 100.
        test_size (int, optional): size of the test set. Defaults to 10.
        random_dataloader (bool, optional): If a random dataloader is used. Defaults to False.

    Returns:
        tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents): Data is a tensor representation of one or more images. Target is an integer representing the images class. Latents are latent representation from the Cats and Dogs classifier.
    """
    torch.manual_seed(seed=random_seed)
    # train the model if not found
    if not os.path.isfile(os.path.join(SAVE_PATH,f"models/model_cad_{cv}.pth")):
        train_model(save_path=SAVE_PATH, cv=cv, random_seed=random_seed)

    classifier = load_model(os.path.join(SAVE_PATH,f"models/model_cad_{cv}.pth"))
    classifier.eval()
    
    test_dir = CAD_TESTDIR
    
    picture_files, labels = get_images(test_dir)
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)
    test_loader = DataLoader(test_set, batch_size=200, shuffle=random_dataloader)
    (test_data, test_targets) = make_corpus(test_loader, corpus_size=test_size, random_seed=random_seed)
    test_data = test_data.detach()
    test_latents = classifier.latent_representation(test_data).detach()

    corpus_dir = CAD_TRAINDIR

    picture_files, labels = get_images(corpus_dir)
    corpus_set = CandDDataSet(image_paths=picture_files, labels=labels)
    corpus_loader = DataLoader(corpus_set, batch_size=200, shuffle=random_dataloader)
    (corpus_data, corpus_target) = make_corpus(corpus_loader=corpus_loader, corpus_size=corpus_size, random_seed=random_seed)
    corpus_data = corpus_data.detach()
    corpus_latents = classifier.latent_representation(corpus_data).detach()

    return classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents)


def train_or_load_heartfailure_model(random_seed: int=42, cv: int=0, corpus_size: int=100, test_size: int=10, random_dataloader: bool=False) -> tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """trains or loads HeartfailureClassifier

    Args:
        random_seed (int, optional): random seed for reproducibility. Defaults to 42.
        cv (int, optional): Cross validation parameter. Defaults to 0.
        corpus_size (int, optional): size of the used corpus. Defaults to 100.
        test_size (int, optional): size of the test set. Defaults to 10.
        random_dataloader (bool, optional): If a random data loader is used. Defaults to False.

    Returns:
        tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents): Data is a tensor representation of one or more input samples. Target is an integer representing the class. Latents are latent representation from the heartfailure classifier.
    """
    torch.manual_seed(random_seed)

    datapath = HEART_FAILURE_DIR

    x,y = load_data(datapath)
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1, random_state=random_seed+cv, shuffle=random_dataloader)
    #trains model if not found
    if not os.path.isfile(os.path.join(SAVE_PATH,"models",f"model_heartfailure_{cv}.pth")):
        train_heartfailure_model(save_path=SAVE_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, cv=cv)

    classifier = HeartFailureClassifier()
    classifier.load_state_dict(torch.load(os.path.join(SAVE_PATH, "models",f"model_heartfailure_{cv}.pth")))
    classifier.eval()

    train_data = HeartFailureDataset(x_train, y_train)
    test_data = HeartFailureDataset(x_test, y_test)
    
    train_loader = DataLoader(train_data, batch_size=50, shuffle=random_dataloader)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=random_dataloader)

    (corpus_data, corpus_target) = make_corpus(train_loader, corpus_size=corpus_size, random_seed=random_seed)

    (test_data, test_targets) = make_corpus(test_loader, corpus_size=test_size, random_seed=random_seed)

    corpus_data = corpus_data.detach()
    test_data = test_data.detach()
    corpus_latents = classifier.latent_representation(corpus_data).detach()
    test_latents = classifier.latent_representation(test_data).detach()

    return classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents)
    