from time import gmtime, strftime
from captum.attr._utils.visualization import visualize_image_attr
import inspect
from math import isclose
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
SAVE_PATH=os.path.join(parentdir, "files")
CAD_TRAINDIR = os.path.join("data","Animal Images","train")
CAD_TESTDIR = os.path.join("data","Animal Images","test")
HEART_FAILURE_DIR = os.path.join("data","heart.csv")

# directly from original code visualization/images.py
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
    fig2 = plot_mnist(corpus_data[most_important_example][0], f"M.i. attribution to test example {test_id} (corpus id {most_important_example})")
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
            title=f"Jacobian of most important corpus example (id {most_important_example}) for test example with id {test_id}",
            use_pyplot=True,
        )
    

def is_close_w_index(a:list[float], b:list[float], tolerance=0.0): 
    """
    source: https://stackoverflow.com/a/72291500/14934164
    compares elements of 2 lists pairwise for numerical diff.
    """
    r=[]
    # same length lists, use zip to iterate pairwise, use enumerate for index
    for idx, (aa, bb) in enumerate(zip(a,b)):
        # convert to floats
        aaa = float(aa)
        bbb = float(bb)

        # append if not close
        if not isclose(aaa,bbb, abs_tol=tolerance):
            r.append((idx, (aaa,bbb)))

    # print results
    for w in r:
        print("On index {0} we have {1} != {2}".format(w[0],*w[1]), sep="\n")
    
    return True if r==[] else False

def get_row_max(a:torch.Tensor) -> list[torch.Tensor]:
    a_max_ids = [int(a[i].argmax()) for i in range(a.shape[0])]
    return a_max_ids

def compare_row_max(a, b):
    diff = []
    for i in range(len(a)):
        diff.append(abs(a[i]-b[i]))
    return diff

def jacobian_compare_score(a:torch.Tensor, b:torch.Tensor):
    a_max = get_row_max(a)
    b_max = get_row_max(b)
    diff = compare_row_max(a_max, b_max)
    return sum(diff)/len(a), a_max, b_max

def create_input_baseline(corpus_shape):
    
    return torch.zeros(corpus_shape)

def plot_results():
    """
    Read results from csv and plot per metric.
    """
    results_df = pd.read_csv(os.path.join(parentdir, "files", "approximation_quality_results_original.csv"))

    decomposition_sizes = [3, 5, 10, 20, 50]

    results_df["explainer"] = results_df["model_type"] + "_" + results_df["dataset"]
    explainer_names = pd.unique(results_df["explainer"])
    explainer_names.sort()

    # settings for plot
    explainer_names = list(explainer_names)
    explainer_names.sort()
    metric_names = ["r2_latent", "r2_output"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    styles = ["--", "--", "-", "--",":", ":", ":", ":"]

    plt.rc("text", usetex=False)
    params = {"text.latex.preamble": r"\usepackage{amsmath}"}
    plt.rcParams.update(params)
    
    sns.set(font_scale=2.0)
    sns.set_style("white")
    sns.set_palette("colorblind")
    mean_df = results_df.groupby(["explainer", "n_keep"]).aggregate("mean", numeric_only=True).unstack(level=0)
    std_df = results_df.groupby(["explainer", "n_keep"]).aggregate(np.std).unstack(level=0)

    # one file per metric (here 2 metrics)
    for m, metric_name in enumerate(metric_names):
        plt.figure(m + 1, figsize=(12,10))
        # add all results to the plot
        for c, s, explainer_name in zip(colors, styles, explainer_names):
            plt.plot(
                decomposition_sizes,
                mean_df[metric_name, explainer_name],
                linestyle=s,
                label=explainer_name,
                color=c,
            )
            # # plots std
            # plt.fill_between(
            #     decomposition_sizes,
            #     mean_df[metric_name, explainer_name] - std_df[metric_name, explainer_name],
            #     mean_df[metric_name, explainer_name] + std_df[metric_name, explainer_name],
            #     alpha=0.2,
            # )

    save_path = os.path.join(SAVE_PATH, "original_experiment")
    timestamp = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    plt.figure(1)
    # plt.ylim((0,1))
    plt.xlabel(r"decomposition size")
    plt.ylabel(r"$R^2_{\mathcal{H}}$")
    plt.legend(loc='lower right', fancybox=True, framealpha=0.5)
    plt.savefig(os.path.join(save_path, f"r2_latent_{timestamp}.pdf"), bbox_inches="tight")
    plt.figure(2)
    plt.ylim((0,1))
    plt.xlabel(r"decomposition size")
    plt.ylabel(r"$R^2_{\mathcal{Y}}$")
    plt.legend(loc='lower right', fancybox=True, framealpha=0.5)
    plt.savefig(os.path.join(save_path, f"r2_output{timestamp}.pdf"), bbox_inches="tight")
