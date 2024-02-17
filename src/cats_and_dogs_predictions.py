import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.simplex_versions import reimplemented_model
from src.utils.image_finder_cats_and_dogs import get_images
from src.utils.corpus_creator import make_corpus
from src.models.CatsAndDogsModel import CatsandDogsClassifier
from src.datasets.cats_and_dogs_dataset import CandDDataSet, LABEL
from src.visualization.images import plot_corpus_decomposition, plot_corpus_decomposition_with_jacobian



def load_model(model_path):
    model = CatsandDogsClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
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

def main():
    model_path = "results\models\model_catsanddogs_ValAcc_0.8826116373477673_Epoch_80.pth"
    model_path2 = "results\models\model_catsanddogs_ValAcc_0.857916102841678_Epoch_20.pth"
    model = load_model(model_path=model_path)
    test_dir = r"data\Animal Images\test"
    picture_files = []

    for root, _, filenames in os.walk(test_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(root, filename)
                last_subfolder = os.path.basename(os.path.dirname(file_path))
                picture_files.append((file_path, LABEL[last_subfolder]))
    
    labels = [label for _, label in picture_files]
    picture_files = [picture_file for picture_file, _ in picture_files]
    
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)

    test_loader = DataLoader(test_set, batch_size=10, shuffle=True)

    for data, label in test_loader:
        for data, label in zip(data, label):
            cat_prob, dog_prob = get_class_probabilites(model,data)
            visualize_image(data, cat_prob=cat_prob, dog_prob=dog_prob)
        break
    
def do_simplex_with_cats_and_dogs(model, corpus_inputs, test_inputs, decomposition_size, test_id):
    corpus_latents = model.latent_representation(corpus_inputs[0].detach()).detach()
    test_latents =  model.latent_representation(test_inputs[0].detach()).detach()
    corpus_inputs = corpus_inputs[0].detach()
    test_inputs = test_inputs[0].detach()
    latent_rep_approximations, weights, jacobian = reimplemented_model(model=model,
                                                                        corpus_inputs=corpus_inputs,
                                                                        corpus_latents=corpus_latents,
                                                                        test_inputs=test_inputs,
                                                                        test_latents=test_latents,
                                                                        decompostion_size=decomposition_size,
                                                                        test_id=test_id)

    return latent_rep_approximations, weights, jacobian


if __name__ == "__main__":
    model_path = "results\models\model_catsanddogs_ValAcc_0.8832882273342354_Epoch_60.pth"
    model = load_model(model_path=model_path)

    test_id=0


    test_dir = r"data\Animal Images\test"

    picture_files, labels = get_images(test_dir)
    
    test_set = CandDDataSet(image_paths=picture_files, labels=labels)

    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    test_inputs = next(iter(test_loader))

    corpus_dir = r"data\Animal Images\train"

    picture_files = []

    picture_files, labels = get_images(corpus_dir)
    
    corpus_set = CandDDataSet(image_paths=picture_files, labels=labels)

    corpus_loader = DataLoader(corpus_set, batch_size=50, shuffle=True)

    #corpus = make_corpus(corpus_loader=corpus_loader)
    
    corpus_inputs = next(iter(corpus_loader))
    print(type(corpus_inputs[0]),len(corpus_inputs), corpus_inputs[0].shape)
    exit()
    corpus_latents = model.latent_representation(corpus_inputs[0].detach()).detach()
    test_latents =  model.latent_representation(test_inputs[0].detach()).detach()
    corpus_inputs = corpus_inputs[0].detach()
    test_inputs = test_inputs[0].detach()

    decomposition_size = 10

    latent_rep_approximations, weights, jacobian = reimplemented_model(model=model,
                                                                        corpus_inputs=corpus_inputs,
                                                                        corpus_latents=corpus_latents,
                                                                        test_inputs=test_inputs,
                                                                        test_latents=test_latents,
                                                                        decompostion_size=decomposition_size,
                                                                        test_id=test_id)
    

    figure = plot_corpus_decomposition(test_image=test_inputs[test_id], corpus=corpus_inputs, weights=weights[test_id])
    plt.show()
    figure = plot_corpus_decomposition_with_jacobian(test_image=test_inputs[0], corpus=corpus_inputs, weights=weights[0], jacobian=jacobian)
    plt.show()
