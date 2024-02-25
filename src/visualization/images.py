import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import torch
from typing import List
from captum.attr._utils.visualization import visualize_image_attr

def plot_corpus_decomposition_with_jacobian(test_image: torch.Tensor, test_pred: str, corpus: torch.Tensor, corpus_preds: List[str], weights: torch.Tensor, jacobian: torch.Tensor, decomposition_length: int=4) -> plt.Figure:
    """Creates a figure with the most important corpus examples and jacobian projection

    Args:
        test_image (torch.Tensor): test image the projection is plotted for
        test_pred (str): string containing the prediction
        corpus (torch.Tensor): tensor with corpus data
        corpus_preds (List[str]): list with strings containing corpus predictions
        weights (torch.Tensor): weights of the corpus data
        jacobian (torch.Tensor): jacobian projection for the corpus data
        decomposition_length (int, optional): how many corpus images shall be printed. Defaults to 2.

    Returns:
        plt.Figure: figure containing test image and top decomposition_length corpus images with jacoiban projection
    """
    if decomposition_length < 1 or decomposition_length > 6 or not(type(decomposition_length)==int): decomposition_length = 6
    if decomposition_length>len(weights): decomposition_length=len(weights)
    sorted_weights_indices = torch.argsort(weights, descending=True)
    sorted_weights = weights[sorted_weights_indices]
    sorted_images = corpus[sorted_weights_indices]
    jacobian = jacobian[sorted_weights_indices]
    fig = plt.figure(layout='constrained')
    gs = gridspec.GridSpec(1,2, figure=fig, width_ratios=[0.2,0.8])

    ax = fig.add_subplot(gs[0])
    ax.imshow(test_image.permute(1,2,0),cmap="gray")
    ax.axis('off')
    ax.set_title(test_pred)
    grid_x_size = (decomposition_length-1)//2+1 

    if decomposition_length==1:
        image = sorted_images[0].numpy().transpose(1,2,0)
        saliency = jacobian[0].numpy().transpose(1,2,0)
        title = f"{corpus_preds[sorted_weights_indices[0]]}, Weigth:Weight:{sorted_weights[0]*100:.2f}%"
        ax = fig.add_subplot(gs[1])
        visualize_image_attr(
                saliency,
                image,
                method="blended_heat_map",
                sign="all",
                plt_fig_axis=(fig, ax),
                title=title,
                alpha_overlay=0.45,
                show_colorbar=False,
                use_pyplot=False
            )
    else:
        gs2 = gridspec.GridSpecFromSubplotSpec(grid_x_size, 2, subplot_spec=gs[1], hspace=0.2)
        index = 0
        for y,x in itertools.product(range(2),range(grid_x_size)):
            if index >=decomposition_length: break
            image = sorted_images[index].numpy().transpose(1,2,0)
            saliency = jacobian[index].numpy().transpose(1,2,0)
            title = f"{corpus_preds[sorted_weights_indices[index]]}, Weight:{sorted_weights[index]*100:.2f}%"
            ax = fig.add_subplot(gs2[x,y])
            visualize_image_attr(
                saliency,
                image,
                method="blended_heat_map",
                sign="all",
                plt_fig_axis=(fig, ax),
                title=title,
                alpha_overlay=0.45,
                show_colorbar=False,
                use_pyplot=False
            )
            index += 1

    return fig

