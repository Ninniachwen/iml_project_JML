import inspect
import os
from sklearn.model_selection import train_test_split
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "")
from original_code.src.simplexai.models.image_recognition import MnistClassifier
from original_code.src.simplexai.experiments import mnist
from src.cats_and_dogs_training import train_model, load_model
from src.classifier.HeartfailureClassifier import HeartFailureClassifier
from src.heartfailure_training import train_heartfailure_model, load_data
from src.datasets.cats_and_dogs_dataset import CandDDataSet
from src.datasets.heartfailure_dataset import HeartFailureDataset
from src.utils.corpus_creator import make_corpus
from src.utils.image_finder_cats_and_dogs import get_images
from src.utils.utlis import CAD_TESTDIR,CAD_TRAINDIR, HEART_FAILURE_DIR

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
SAVE_PATH=os.path.join(parentdir, "files", "classifier")


def train_or_load_mnist(random_seed:int = 42, cv: int = 0, corpus_size:int=100, test_size:int=10, random_dataloader: bool=False, use_corpus_maker=False) -> tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Loads the mnist classifier if one is stored or trains one instead. Also loads the mnist dataset

    Args:
        random_seed (int, optional): random seed for seeding the mnist model. Defaults to 42.
        cv (int, optional): is added to random seed 42, to run experiments with different random seeds. used as identifier for stored models. Defaults to 0.
        corpus_size (int, optional): How many images to use for corpus (explainer images). Defaults to 100.
        test_size (int, optional): How many images to use for test_set (images that will be explained, using corpus). Defaults to 10.
        random_dataloader (bool, optional): Whether samples random images or the same images in each run. Defaults to False.
        use_corpus_maker (bool, optional): Whether to use the corpus maker (randomizing data). Defaults to False.


    Returns:
        tuple[MnistClassifier, tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents): Data is a tensor representation of one or more images. Target is an integer representing the images class. Latents are latent representation from the mnist classifier.
    """
    # train and store classifier, if none exists
    classifier_name = f"classifier_mnist_cv{cv}.pth"
    if not os.path.isfile(os.path.join(SAVE_PATH,classifier_name)):
        mnist.train_model( # using standard values
                device="cpu",
                random_seed=random_seed,
                cv=cv,
                save_path=SAVE_PATH,
                model_reg_factor=0.1,
            )
    
    # load classifier
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(SAVE_PATH,classifier_name)))
    classifier.eval()

    # data loader from approximate_quality of mnist.py of original code
    corpus_loader = mnist.load_mnist(corpus_size, train=True, shuffle=random_dataloader)
    test_loader = mnist.load_mnist(test_size, train=False, shuffle=random_dataloader)
    if use_corpus_maker:
        corpus_data, corpus_target = make_corpus(corpus_loader, corpus_size=corpus_size, n_classes=10, random_seed=random_seed+cv)
        test_data, test_targets = make_corpus(test_loader, corpus_size=corpus_size, n_classes=10, random_seed=random_seed+cv)
    else: # use simplex dataloader
        batch_id_corpus, (corpus_data, corpus_target) = next(enumerate(corpus_loader))
        corpus_data = corpus_data.detach()
        batch_id_test, (test_data, test_targets) = next(enumerate(test_loader))
        test_data = test_data.detach()
    
    # get latent representations from classifier
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
    # train the model if not found
    filename = f"classifier_cad_{cv}.pth"
    if not os.path.isfile(os.path.join(SAVE_PATH, filename)):
        train_model(save_path=SAVE_PATH, cv=cv, random_seed=random_seed)

    classifier = load_model(os.path.join(SAVE_PATH, filename))
    classifier.eval()
    
    test_dir = CAD_TESTDIR
    
    picture_files, labels = get_images(test_dir)
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)
    test_loader = DataLoader(test_set, batch_size=200, shuffle=random_dataloader)
    (test_data, test_targets) = make_corpus(test_loader, corpus_size=test_size, random_seed=random_seed+cv)
    test_data = test_data.detach()
    test_latents = classifier.latent_representation(test_data).detach()

    corpus_dir = CAD_TRAINDIR

    picture_files, labels = get_images(corpus_dir)
    corpus_set = CandDDataSet(image_paths=picture_files, labels=labels)
    corpus_loader = DataLoader(corpus_set, batch_size=200, shuffle=random_dataloader)
    (corpus_data, corpus_target) = make_corpus(corpus_loader=corpus_loader, corpus_size=corpus_size, random_seed=random_seed+cv)
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

    datapath = HEART_FAILURE_DIR

    x,y = load_data(datapath)
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1, random_state=random_seed+cv, shuffle=random_dataloader)
    #trains model if not found
    filename = f"classifier_heartfailure_{cv}.pth"
    if not os.path.isfile(os.path.join(SAVE_PATH, filename)):
        train_heartfailure_model(save_path=SAVE_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, cv=cv)

    classifier = HeartFailureClassifier()
    classifier.load_state_dict(torch.load(os.path.join(SAVE_PATH,filename)))
    classifier.eval()

    train_data = HeartFailureDataset(x_train, y_train)
    test_data = HeartFailureDataset(x_test, y_test)
    
    train_loader = DataLoader(train_data, batch_size=50, shuffle=random_dataloader)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=random_dataloader)

    (corpus_data, corpus_target) = make_corpus(train_loader, corpus_size=corpus_size, random_seed=random_seed+cv)

    (test_data, test_targets) = make_corpus(test_loader, corpus_size=test_size, random_seed=random_seed+cv)

    corpus_data = corpus_data.detach()
    test_data = test_data.detach()
    corpus_latents = classifier.latent_representation(corpus_data).detach()
    test_latents = classifier.latent_representation(test_data).detach()

    return classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents)
    