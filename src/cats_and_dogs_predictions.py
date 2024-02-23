import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.simplex_versions import reimplemented_model
from src.utils.image_finder_cats_and_dogs import get_images
from src.utils.corpus_creator import make_corpus
from src.classifier.CatsAndDogsClassifier import CatsandDogsClassifier
from src.datasets.cats_and_dogs_dataset import CandDDataSet, LABEL
from src.cats_and_dogs_training import get_classes_for_preds
from src.visualization.images import plot_corpus_decomposition, plot_corpus_decomposition_with_jacobian

def load_model(model_path):
    """
    Loads model from path. 
    """
    model = CatsandDogsClassifier()
    model.load_state_dict(torch.load(model_path))
    return model


def get_class_probabilites(model, x):
    """
    Computes class probabilites from model output
    """
    pred = model(x).detach()
    pred *= 100
    return (100-pred).item(),pred.item()

def visualize_image(x, cat_prob, dog_prob):
    """
    Plots an image with corresponding class probability
    :param x: torch tensor of shape (1,150,150)
    :param cat_prob: probability of class 0 (cats)
    :param dog_prob: probability of class 1 (dogs)
    """
    plt.imshow(x.permute(1,2,0),cmap="gray")
    plt.xlabel(f"Cat prob: {cat_prob:.2f}%, Dog prob: {dog_prob:.2f}%")
    plt.show()

if __name__ == "__main__":
    model_path = "files\models\model_cad_0.pth"
    model = load_model(model_path=model_path)

    test_id=0


    test_dir = r"data\Animal Images\test"

    picture_files, labels = get_images(test_dir)
    
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    test_inputs = next(iter(test_loader))

    corpus_dir = r"data\Animal Images\train"

    picture_files = []

    picture_files, labels = get_images(corpus_dir)
    
    corpus_set = CandDDataSet(image_paths=picture_files, labels=labels)

    corpus_loader = DataLoader(corpus_set, batch_size=5, shuffle=True)

    corpus_inputs = make_corpus(corpus_loader=corpus_loader, corpus_size=500)
    
    corpus_latents = model.latent_representation(corpus_inputs[0])
    test_latents =  model.latent_representation(test_inputs[0])
    corpus_inputs = corpus_inputs[0]
    test_inputs = test_inputs[0]

    decomposition_size = 10 
    print((test_inputs[test_id]).shape)
    pred = model(test_inputs)
    corpus_pred = model(corpus_inputs)
    pred = get_classes_for_preds(pred)
    corpus_pred = get_classes_for_preds(corpus_pred)
    print(pred, corpus_pred)

    latent_rep_approximations, weights, jacobian = reimplemented_model(classifier=model,
                                                                        corpus_inputs=corpus_inputs,
                                                                        corpus_latents=corpus_latents,
                                                                        test_data=test_inputs,
                                                                        test_latents=test_latents,
                                                                        decompostion_size=decomposition_size,
                                                                        test_id=test_id,
                                                                        input_baseline=torch.zeros(corpus_inputs.shape))
    

    figure = plot_corpus_decomposition(test_image=test_inputs[test_id], test_pred=pred[0], corpus=corpus_inputs, corpus_preds=corpus_pred, weights=weights[test_id])
    plt.show()
    figure = plot_corpus_decomposition_with_jacobian(test_image=test_inputs[0], test_pred=pred[0],corpus=corpus_inputs, corpus_preds=corpus_pred, weights=weights[0], jacobian=jacobian, decomposition_length=4)
    plt.show()
