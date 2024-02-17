import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import torch
import torchvision.transforms as t

def plot_corpus_decomposition(test_image, corpus, weights, title: str = "") -> plt.Figure:
    """
    Plots test image and corpus decomposition. Gives maximally 6 corpus images and jacobian coloration
    """
    decomposition_length = len(weights)
    if decomposition_length > 6: decomposition_length = 6
    sorted_weights_indices = torch.argsort(weights, descending=True)
    print(weights.sum())
    sorted_weights = weights[sorted_weights_indices]
    sorted_images = corpus[sorted_weights_indices]

    fig = plt.figure(layout='constrained')

    gs = gridspec.GridSpec(1,2, figure=fig, width_ratios=[0.3,0.7])

    ax = fig.add_subplot(gs[0])
    ax.imshow(test_image.permute(1,2,0),cmap="grey")
    ax.axis('off')
    ax.set_title('Test Image')
    print(sorted_weights.sum())
    grid_x_size = (decomposition_length-1)//2+1 


    gs2 = gridspec.GridSpecFromSubplotSpec(grid_x_size, 2, subplot_spec=gs[1])
    index = 0
    for y,x in itertools.product(range(2),range(grid_x_size)):
        if index >=len(sorted_images): break

        ax = fig.add_subplot(gs2[x,y])
        ax.imshow(sorted_images[index].permute(1,2,0), cmap="grey")
        ax.axis('off')
        ax.set_title(f"{sorted_weights[index].detach().item()*100:.2f}%")
        index += 1
    

    return fig


def plot_corpus_decomposition_with_jacobian(test_image, corpus, weights, jacobian, title: str = "") -> plt.Figure:
    """
    Plots test image and corpus decomposition. Gives maximally 5 corpus images and jacobian coloration
    """
    decomposition_length = len(weights)#TODO: recompostionlength=1?
    if decomposition_length > 6: decomposition_length = 6
    sorted_weights_indices = torch.argsort(weights, descending=True)
    
    sorted_weights = weights[sorted_weights_indices]
    sorted_images = corpus[sorted_weights_indices]
    jacobian = jacobian[sorted_weights_indices]
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
        if index >=len(sorted_images): break
        ax = fig.add_subplot(gs2[x,y])
        ax.imshow(sorted_images[index].permute(1,2,0), cmap="grey")
        #print(jacobian[index], jacobian[index].shape)
        #print( "min",jacobian[index].min(1, keepdim=True)[0])
        #jacobian[index] -= jacobian[index].min(1, keepdim=True)[0]
        #print(jacobian[index])
        #jacobian[index] /= jacobian[index].max(1, keepdim=True)[0]
        #jacobian[index] = torch.nn.functional.log_softmax(jacobian[index], dim=0)
        #print(jacobian[index])
        ax.imshow(jacobian[index].permute(1,2,0),cmap='RdYlGn', alpha=1)
        ax.axis('off')
        ax.set_title(f"{sorted_weights[index].detach().item()*100:.2f}%")
        index += 1
    
    
    
    return fig