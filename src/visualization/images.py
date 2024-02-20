import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import torch
import torchvision.transforms as t
from captum.attr._utils.visualization import visualize_image_attr

def plot_corpus_decomposition(test_image, corpus, weights, decomposition_length=0, title: str = "") -> plt.Figure:
    """
    Plots test image and corpus decomposition. Gives maximally 6 corpus images
    """
    if decomposition_length < 1 or decomposition_length > 6 or not(type(decomposition_length)==int) or decomposition_length>len(weights): decomposition_length = 6
    sorted_weights_indices = torch.argsort(weights, descending=True)
    sorted_weights = weights[sorted_weights_indices]
    sorted_images = corpus[sorted_weights_indices]

    fig = plt.figure(layout='constrained')

    gs = gridspec.GridSpec(1,2, figure=fig, width_ratios=[0.3,0.7])

    ax = fig.add_subplot(gs[0])
    ax.imshow(test_image.permute(1,2,0),cmap="grey")
    ax.axis('off')
    ax.set_title('Test Image')
    grid_x_size = (decomposition_length-1)//2+1 


    gs2 = gridspec.GridSpecFromSubplotSpec(grid_x_size, 2, subplot_spec=gs[1])
    index = 0
    for y,x in itertools.product(range(2),range(grid_x_size)):
        if index >=len(sorted_images) | index >decomposition_length: break

        ax = fig.add_subplot(gs2[x,y])
        ax.imshow(sorted_images[index].permute(1,2,0), cmap="grey")
        ax.axis('off')
        ax.set_title(f"{sorted_weights[index].detach().item()*100:.2f}%")
        index += 1
    

    return fig


def plot_corpus_decomposition_with_jacobian(test_image, save_path, corpus, weights, jacobian, test_id, decomposition_length=0, title: str = "") -> plt.Figure:
    """
    Plots test image and corpus decomposition. Gives maximally 6 corpus images and jacobian coloration
    """
    if decomposition_length > 6 | decomposition_length>len(corpus): decomposition_length = 6
    sorted_weights_indices = torch.argsort(weights, descending=True)
    
    sorted_weights = weights[sorted_weights_indices]
    sorted_images = corpus[sorted_weights_indices]
    jacobian = jacobian[sorted_weights_indices]
    fig = plt.figure(layout='constrained')

    gs = gridspec.GridSpec(1,2, figure=fig, width_ratios=[0.2,0.8])

    ax = fig.add_subplot(gs[0])
    ax.imshow(test_image.permute(1,2,0),cmap="grey")
    ax.axis('off')
    ax.set_title('Test Image')
    grid_x_size = (decomposition_length-1)//2+1 


    gs2 = gridspec.GridSpecFromSubplotSpec(grid_x_size, 2, subplot_spec=gs[1], hspace=0.2)
    index = 0
    for y,x in itertools.product(range(2),range(grid_x_size)):
        if index >=decomposition_length: break
        image = sorted_images[index].numpy().transpose(1,2,0)
        saliency = jacobian[index].numpy().transpose(1,2,0)
        title = f"Weight: {sorted_weights[index]*100:.2f}%"
        ax = fig.add_subplot(gs2[x,y])
        visualize_image_attr(#TODO: finish
            saliency,
            image,
            method="blended_heat_map",
            sign="all",
            plt_fig_axis=(fig, ax),
            title=title,
            alpha_overlay=0.6,
            show_colorbar=True,
            use_pyplot=False
        )
        index += 1
    
    return fig

