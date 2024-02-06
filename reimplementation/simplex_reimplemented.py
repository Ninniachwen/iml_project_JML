from captum.attr._utils.visualization import visualize_image_attr
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt

#TODO: from simplex
class Scheduler:
    def __init__(self, n_epoch):
        self.n_epoch = n_epoch
#TODO: from simplex        
class ExponentialScheduler(Scheduler):
    def __init__(self, x_init: float, x_final: float, n_epoch: int):
        Scheduler.__init__(self, n_epoch)
        self.step_factor = math.exp(math.log(x_final / x_init) / n_epoch)

    def step(self, x):
        return x * self.step_factor
#TODO: from simplex
def plot_mnist(data, title: str = "") -> plt.Figure:
    fig = plt.figure()
    plt.imshow(data, cmap="gray", interpolation="none")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return fig
#TODO: from simplex

# from https://nextjournal.com/gkoehler/pytorch-mnist
class Mnist_Model(nn.Module):
    def __init__(self):
        super(Mnist_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def latent(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x)
        return x
    
    def dataloader(self, train, batch_size):
        mean_mnist = 0.1307
        std_dev_mnist = 0.1307
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                '/files/',
                train=train,
                download=True, 
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (mean_mnist,), (std_dev_mnist,))
                    ])),    #TODO: different or no transformations?
                batch_size=batch_size,
                shuffle=True)
        
def plot_data(data_loader):
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)


    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig.show()


def simplex_re(model_path:str, optimizer_path:str, decompostion_size = 5, test_size=1, corpus_size=1000, epochs = 10000, reg_factor=1.0, test_id=22):
    torch.manual_seed(seed=42)

    # load Model
    classifier = Mnist_Model()
    classifier.load_state_dict(torch.load(model_path))
    #classifier.to("cpu")
    classifier.eval()

    # prepare data
    corpus_loader = classifier.dataloader(train=True, batch_size=corpus_size)
    _, (corpus_data, _) = next(enumerate(corpus_loader))
    test_loader = classifier.dataloader(train=False, batch_size=test_size)
    _, (test_data, _) = next(enumerate(test_loader))
    if test_id < test_size:
        test_data_point = test_data[test_id : test_id + 1].detach()    #selects a single test-datapoint
    else:
        test_data_point = test_data
    #plot_data(test_loader)

    # 
    corpus_latent = classifier.latent(corpus_data).detach()
    test_latent = classifier.latent(test_data_point).detach()

    scheduler = ExponentialScheduler(x_init=0.1, x_final=1000, n_epoch=20000)
    
    #simplex.fit
    W_0 = torch.zeros((test_size, corpus_size), requires_grad=True)
    optimizer = torch.optim.Adam([W_0])
    
    for epoch in range(epochs):
        #TODO: from simplex
        optimizer.zero_grad()
        weights = F.softmax(W_0, dim=-1)
        corpus_latent = torch.einsum(
            "ij,jk->ik", weights, corpus_latent
        )
        error = ((corpus_latent - test_latent) ** 2).sum()
        weights_sorted = torch.sort(weights)[0]
        regulator = (weights_sorted[:, : (corpus_size - decompostion_size)]).sum()
        loss = error + reg_factor * regulator
        loss.backward()
        optimizer.step()
        if (epoch + 1) % (epochs / 5) == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{epochs} ; Error: {error.item():.3g} ;"
                f" Regulator: {regulator.item():.3g} ; Reg Factor: {reg_factor:.3g}"
            )
        reg_factor = scheduler.step(reg_factor)

        #TODO: from simplex
    # end of fit
    weights = torch.softmax(W_0, dim=-1).detach()
    
    assert test_id < test_size
    weights = weights[test_id].numpy()
    sort_id = np.argsort(weights)[::-1]
    
    corpus_decomposition = []
    for i in sort_id:
        corpus_decomposition.append((weights[i], corpus_data[i]))
    

    output = classifier(test_data_point)
    title = f"Prediction: {output.data.max(1, keepdim=True)[1][0].item()}"
    fig = plot_mnist(test_data_point[0][0].numpy(), title)
    fig.savefig(f"test_image_id{test_id}")
    fig.show()

    for i in range(decompostion_size):
        image = corpus_decomposition[i][1].cpu().numpy().transpose((1, 2, 0))
        title = f"Weight: {corpus_decomposition[i][0]:.2g}"
        fig = plot_mnist(image[0][0].numpy(), title)
        # fig, axis = visualize_image_attr(
        #     image,
        #     method="blended_heat_map",
        #     sign="all",
        #     title=title,
        #     use_pyplot=True,
        # )
        fig.savefig(f"corpus_image{i + 1}_id{test_id}")
        fig.show()

    print()

    #x = self.fc2(x)
    #return F.log_softmax(x)


if __name__ == "__main__":
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "iml", "project-jml-project", "reimplementation", "model.pth")
    optimizer_path = os.path.join(cwd, "iml", "project-jml-project", "reimplementation",  "optimizer.pth")

    simplex_re(model_path, optimizer_path)