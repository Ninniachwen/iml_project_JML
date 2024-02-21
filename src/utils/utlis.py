# directly from original code visualization/images.py
from matplotlib import pyplot as plt
import torch
from captum.attr._utils.visualization import visualize_image_attr


def plot_mnist(data:torch.tensor, title: str = "") -> plt.Figure:
    """
    Plot given data as image.

    Args:
        data (torch.tensor): tensor with image data.
        title (str, optional): title for the matplot figure. Defaults to "".

    Returns:
        plt.Figure: figure showing the image
    """
    fig = plt.figure()
    plt.imshow(data, cmap="gray", interpolation="none")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return plt


def plot_test_img_and_most_imp_explainer(weights:torch.tensor, corpus_data:torch.tensor, test_data:torch.tensor, test_id:int):
    """
    Plots the test image we want to explain and the most important corpus example for the explanation. 

    Args:
        weights (torch.tensor): weights of the corpus examples
        corpus_data (torch.tensor): corpus data tensors
        test_data (torch.tensor): test data tensors
        test_id (int): Id of the test image we want to explain
    """
    most_important_example = weights[test_id].argmax()  # for test example 0
    fig1 = plot_mnist(test_data[test_id][0], f"Test example {test_id}")
    fig2 = plot_mnist(corpus_data[most_important_example][0], f"M.i. attribution to example 0 (corpus id {most_important_example})")
    fig1.show()
    fig2.show()

    print(f"Biggest contributor to test example {test_id}: {weights[0].argmax()} with weight {weights[0].max()}")

def plot_jacobians_grayscale(jacobian:torch.Tensor): #shape 28,28 for MNIST
    """
    Plots given jacobian tensor (saliencey) as grayscale.

    Args:
        jacobian (torch.Tensor): jacobian projection as tensor
    """
    fig2 = plot_mnist(jacobian)
    fig2.show()

def print_jacobians_with_img(weights:torch.tensor, test_id:int, corpus_data:torch.tensor, jacobian:torch.tensor) -> None:
    """
    Prints jacobians (saliency) with original image.
    Args:
        weights (torch.tensor): weights of the corpus examples
        test_id (int): Id of the test image we want to explain
        corpus_data (torch.tensor): corpus data tensors
        jacobian (torch.tensor): jacobian projection as tensor
    """
   
    most_important_example = weights[test_id].argmax()
    image = corpus_data[most_important_example].numpy().transpose((1, 2, 0))  # transpose, see use_case.py from original code
    saliency = jacobian[most_important_example].numpy().transpose((1, 2, 0))  # transpose, see use_case.py from original code
    # the following code, see use_case.py from original code
    fig3, _ = visualize_image_attr(
            saliency,
            image,
            method="blended_heat_map",
            sign="all",
            title="Jacobian of most important corpus example",
            use_pyplot=True,
        )