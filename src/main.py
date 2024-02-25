import argparse
import csv
import enum
import inspect
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from time import time, strftime, gmtime
import torch
#TODO: check if requirements file is sufficient

sys.path.insert(0, "")
import src.evaluation as e
import src.simplex_versions as s
import src.classifier_versions as c
from src.visualization.images import plot_corpus_decomposition_with_jacobian
from src.utils.utlis import create_input_baseline, print_jacobians_with_img, plot_test_img_and_most_imp_explainer

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
SAVE_PATH=os.path.join(parentdir, "files")
RANDOM_SEED=42

class Model_Type(enum.Enum):
    """
    List of possible simplex models. This defines what model is used and what settings are set (like softmax, normalize, no regularization, ...)
    """
    ORIGINAL = 1  # use original model grom github repository
    ORIGINAL_COMPACT = 2  # use original code which was rewritten to fit in one function
    REIMPLEMENTED = 3  # use reimplemented model
    R_NO_SOFTMAX = 4  # use reimplemented model but without softmax layer during training (for ablation)
    R_NORMALIZE = 5  # use reimplemented model but using a normalization layer during training (for ablation)
    R_INIT_WEIGHTS_RANDOM = 6 # use reimplemented model but use set initial weights with random values (for ablation)
    R_NO_SOFTMAX_IWR = 7  # like R_NO_SOFTMAX but use set initial weights with random values (for ablation)
    R_NORMALIZE_IWR = 8  # like R_NORMALIZE but use set initial weights with random values (for ablation)
    O_C_NO_SOFTMAX = 9  # use compact original (ORIGINAL_COMPACT) but without softmax layer during training (for ablation)
    O_C_NO_REGULIZATION = 10  # use compact original (ORIGINAL_COMPACT) but use no regularization during training (for ablation)

class Dataset(enum.Enum):
    """
    Datasets that can be used in this setup. Each dataset comes with a classifier and dataloader.
    """
    MNIST = 1
    CaD = 2
    Heart = 3
    MNIST_MakeCorpus = 4

def do_simplex(model_type=Model_Type.ORIGINAL, dataset=Dataset.MNIST, cv=0, decomposition_size=100, corpus_size=100, test_size=10, test_id=0, print_jacobians=False, print_test_example=False, r_2_scores=True, decompose=True, random_dataloader=False) -> tuple[torch.Tensor, None|list[float], None|list[float], torch.Tensor, None|list[dict]]|None:
    """
    Decide which simplex model we want to train with which dataset.

    Args:
        model_type (_type_, optional): The type of simplex model; see Model_Type Class above. Defaults to Model_Type.ORIGINAL.
        dataset (_type_, optional): which dataset to use; see Dataset Class above. Defaults to Dataset.MNIST.
        cv (int, optional): _description_. Defaults to 0.
        decomposition_size (int, optional): with how many corpus examples a test example should be explained. Defaults to 100.
        corpus_size (int, optional): how many examples should be in the corpus. Defaults to 100.
        test_size (int, optional): how many test examples we want to explain. Defaults to 10.
        test_id (int, optional): what id of the test examples we want to explain. Defaults to 0.
        print_jacobians (bool, optional): if we want to print the jacobians. Defaults to False.
        print_test_example (bool, optional): if we want to print the test example and the according most important corpus example. Defaults to False.
        r_2_scores (bool, optional): if we want to return the r2 scores. Defaults to True.
        decompose (bool, optional): if we want to create a decomposition (see evaluation.py). Defaults to True.
        random_dataloader (bool, optional): if we want to load the data randomly. Defaults to False.

    Raises:
        Exception 1: invalid dataset: if dataset is not part of Dataset Enum.
        Exception 2: if corpus_size is smaller than decomposition_size.
        Exception 3: if test_size is smaller or equal to test_id.

    Returns:
        tuple[torch.Tensor, None|list[float], None|list[float], torch.Tensor, None|list[dict]]|None: weights, latent_r2_score, output_r2_score, jacobian, decompostions
    """

    assert corpus_size >= decomposition_size, "decomposition size can't be larger than corpus"
    assert test_size > test_id, "test_id can't be larger than test_size"

    torch.random.manual_seed(RANDOM_SEED + cv)
    torch.backends.cudnn.deterministic = True 
    # https://www.typeerror.org/docs/pytorch/generated/torch.nn.conv1d

    # must return classifier, (corpus_data, corpus_target, corpus_latents), (test_data, test_targets, test_latents)
    if dataset is Dataset.MNIST:
        classifier, corpus, test_set = c.train_or_load_mnist(RANDOM_SEED, cv, corpus_size=corpus_size, test_size=test_size, random_dataloader=random_dataloader)
        corpus_data, corpus_target, corpus_latents = corpus
        test_data, test_targets, test_latents = test_set
        
    elif dataset is Dataset.CaD:
        classifier, corpus, test_set = c.train_or_load_CaD_model(RANDOM_SEED, cv, corpus_size=corpus_size, test_size=test_size, random_dataloader=random_dataloader)
        corpus_data, corpus_target, corpus_latents = corpus
        test_data, test_targets, test_latents = test_set
        
    elif dataset is Dataset.Heart:
        classifier, corpus, test_set = c.train_or_load_heartfailure_model(RANDOM_SEED, cv, corpus_size=corpus_size, test_size=test_size, random_dataloader=random_dataloader)
        corpus_data, corpus_target, corpus_latents = corpus
        test_data, test_targets, test_latents = test_set

    elif dataset is Dataset.MNIST_MakeCorpus:
        classifier, corpus, test_set = c.train_or_load_mnist(RANDOM_SEED, cv, corpus_size=corpus_size, test_size=test_size, random_dataloader=random_dataloader, use_corpus_maker=True)
        corpus_data, corpus_target, corpus_latents = corpus
        test_data, test_targets, test_latents = test_set
        
    else:
        raise Exception(f"'{dataset}' is no valid input for dataset")
    
    input_baseline = create_input_baseline(corpus_data.shape)

    # must return latent_rep_approx, weights, jacobian
    if model_type is Model_Type.ORIGINAL:
        print(f"Starting on cv {cv} with the original model!")
        latent_rep_approx, weights, jacobian = s.original_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline)

    elif model_type is Model_Type.ORIGINAL_COMPACT:
        print(f"Starting on cv {cv} with the compact original model!")
        latent_rep_approx, weights, jacobian = s.compact_original_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline)
    
    elif (model_type is Model_Type.O_C_NO_SOFTMAX) & (dataset is Dataset.MNIST):
        print(f"Starting on cv {cv} with the compact original model but without using softmax layer while training!")
        latent_rep_approx, weights, jacobian = s.compact_original_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline, softmax=False)
        
    elif (model_type is Model_Type.O_C_NO_REGULIZATION) & (dataset is Dataset.MNIST):
        print(f"Starting on cv {cv} with the compact original model but without using regularization while training!")
        latent_rep_approx, weights, jacobian = s.compact_original_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline, regularisation=False)
        
    elif model_type is Model_Type.REIMPLEMENTED:
        print(f"Starting on cv {cv} with our own reimplemented model!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline)
    
    elif (model_type is Model_Type.R_NORMALIZE) & (dataset is Dataset.MNIST):
        print(f"Starting on cv {cv} with the compact original model but using normalize instead of softmax layer!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline, mode="normalize")
        
    elif (model_type is Model_Type.R_NO_SOFTMAX) & (dataset is Dataset.MNIST):
        print(f"Starting on cv {cv} with our own reimplemented model but without using softmax layer!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline, mode="nothing")

    elif (model_type is Model_Type.R_INIT_WEIGHTS_RANDOM) & (dataset is Dataset.MNIST):
        print(f"Starting on cv {cv} with our own reimplemented model but initializing weights at random!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline, weight_init_zero=False)
    
    elif (model_type is Model_Type.R_NORMALIZE_IWR) & (dataset is Dataset.MNIST):
        print(f"Starting on cv {cv} with the compact original model but using normalize instead of softmax layer and initializing weights at random!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline, mode="normalize", weight_init_zero=False)
        
    elif (model_type is Model_Type.R_NO_SOFTMAX_IWR) & (dataset is Dataset.MNIST):
        print(f"Starting on cv {cv} with our own reimplemented model but without using softmax layer and initializing weights at random!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, input_baseline, mode="nothing", weight_init_zero=False)
        
    else:
        print(f"skipping '{model_type}'&'{dataset}' is no valid combination or no valid input for model_type")
        return None
    
    latent_rep_true = test_latents
    
    latent_r2_score = None
    output_r2_score = None
    if r_2_scores:
        output_r2_score, latent_r2_score = e.r_2_scores(classifier, latent_rep_approx, latent_rep_true)
    
    decompostions = None
    if decompose:
        decompostions = e.create_decompositions(test_data, test_targets, corpus_data, corpus_target, decomposition_size, weights, model_type, dataset)

    if print_jacobians and dataset != Dataset.Heart:
        print_jacobians_with_img(weights, test_id, corpus_data, jacobian)

    if print_test_example and dataset != Dataset.Heart:
        plot_test_img_and_most_imp_explainer(weights, corpus_data, test_data, test_id)
        
    
    return weights, latent_r2_score, output_r2_score, jacobian, decompostions, test_data, corpus_data, classifier


def run_all_experiments(corpus_size=100, test_size=10, decomposition_size=3, cv=0, test_id=0, filename="comparison_results.csv", random_dataloader=False, no_ablation=True, plot_decomposition=True, datasets:list=list(Dataset)) -> tuple[list[torch.Tensor], list[list[float]], list[list[float]], list[torch.Tensor], list[list[dict]]]:#TODO update
    """
    runs all experiments over different sets of models and datasets, depending on settings.
    no_ablation=True: first 3 simplex models, and all 4 datasets.  no_ablation=False: like ablation study, over all simplex models, including ablation versions, but only on mnist dataset.

    Args:
        corpus_size (int, optional): How many images to use for corpus (explainer images). Defaults to 100.
        test_size (int, optional): How many images to use for test_set (images that will be explained, using corpus). Defaults to 10.
        decomposition_size (int, optional): Top x images to be chosen from corpus to create the final decomposition (expanation). Also influences regularization and weights. Defaults to 3.
        cv (int, optional): is added to random seed 42, to run experiments with different random seeds. used as identifier for stored models. Defaults to 0.
        test_id (int, optional): test image to choose from test set when prints are done. eg for jacobians. Defaults to 0.
        filename (str, optional): filename to use for the results of this round of experiments. Defaults to "comparison_results.csv".
        random_dataloader (bool, optional): Whether samples random images or the same images in each run. Defaults to False.
        plot_decomposition (bool, optional): whether the created decompositions are plotted. Defaults to True.
        datasets (list|None, optional): a list of selected datasets from enum Dataset. Defaults to a list of all datasets from the enum.

    Returns:
        tuple[list[torch.Tensor], list[list[float]], list[list[float]], list[torch.Tensor], list[list[dict]]]: returns weights_all, latent_r2_scores, output_r2_scores, jacobians, decompostions. 
    """
    print(f"   starting test runs with parameters:\n   corpus_size: {corpus_size}, test_size: {test_size}, decomposition_size: {decomposition_size}, cv: {cv}, test_id: {test_id}")
              
    weights_all = []
    latent_r2_scores = []
    output_r2_scores = []
    jacobians = []
    decompostions = []

    file_path = os.path.join(SAVE_PATH , filename)
    file = Path(file_path)
    
    mode = "a" if file.is_file() else "w"   # append if file exists, assuming either both files exist, or none
    with open(file_path, mode) as f1:
    
        writer=csv.writer(f1, delimiter=";",lineterminator="\n",)

        if mode == "w":     # add header if file new
            writer.writerow([
                        "timestamp",
                        "corpus_size",
                        "test_size",
                        "decomposition_size",
                        "cv",
                        "model_type",
                        "dataset",
                        "latent_r2_score",
                        "output_r2_score",
                        "sample_id",
                        "target",
                        "most_imp_corpus_id",
                        "most_imp_corpus_weight",
                        "most_imp_corpus_target",
                        "corpus_ids",
                        "corpus_weight",
                        "corpus_targets",
                        "visualization"
                        ])
            
        models = list(Model_Type)[:3] if no_ablation else Model_Type
        if not datasets:
            datasets = list(Dataset) if no_ablation else [list(Dataset)[0]]
        for d in datasets:
            for m in models:
                print(f"   model: {m}, dataset: {d}")
                weights, latent_r2_score, output_r2_score, jacobian, decompostion, test_data, corpus_data, classifier = do_simplex(
                    model_type=m, 
                    dataset=d, 
                    cv=cv,
                    corpus_size=corpus_size, 
                    test_size=test_size, 
                    decomposition_size=decomposition_size, 
                    test_id=test_id,
                    random_dataloader=random_dataloader)
                
                weights_all.append(weights)
                latent_r2_scores.append(latent_r2_score)
                output_r2_scores.append(output_r2_score)
                jacobians.append(jacobian)
                decompostions.append(decompostion)

                dec_c = decompostion[test_id]["decomposition"]
                corpus_ids = [dec_c[i]["c_id"] for i in range(decomposition_size)]
                corpus_weights = [dec_c[i]["c_weight"] for i in range(decomposition_size)]
                corpus_targest = [dec_c[i]["c_target"] for i in range(decomposition_size)]
                time_stamp = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
                image_save_path = "No visualization"
                if plot_decomposition:
                    if d == Dataset.CaD:
                        test_pred = classifier(test_data)
                        test_pred = f"Cat: {(1-test_pred[test_id].item())*100:.2f}%" if test_pred[test_id].item()<0.5 else f"Dog: {(test_pred[test_id].item())*100:.2f}%"
                        corpus_pred = []
                        corpus_preds = classifier(corpus_data)
                        for pred in corpus_preds:
                            if pred <0.5:
                                corpus_pred.append(f"Cat: {(1-pred.item()) * 100:.2f}%")
                            else:
                                corpus_pred.append(f"Dog: {(pred.item() * 100):.2f}%")
                        figure = plot_corpus_decomposition_with_jacobian(test_image=test_data[test_id],
                        test_pred=test_pred,
                        corpus=corpus_data,
                        corpus_preds=corpus_pred,
                        weights=weights[test_id],
                        jacobian=jacobian,
                        decomposition_length=decomposition_size)
                        image_save_path = os.path.join("files","images",f"{d}_{cv}_{test_id}_{m}.png")
                        figure.savefig(image_save_path)
                    elif d in [Dataset.MNIST,Dataset.MNIST_MakeCorpus]:
                        test_pred = classifier.probabilities(test_data)
                        pred = torch.argmax(test_pred, dim=1)
                        test_pred = f"{pred[test_id]}: {(test_pred[test_id][pred[test_id]]) * 100:.2f}%"
                        corpus_pred = []
                        corpus_preds = classifier.probabilities(corpus_data)
                        for c_pred in corpus_preds:
                            max_arg = torch.argmax(c_pred, dim=0)
                            corpus_pred.append(f"{max_arg}: {(c_pred[max_arg] * 100):.2f}%")
                        figure = plot_corpus_decomposition_with_jacobian(test_image=test_data[test_id],
                        test_pred=test_pred,
                        corpus=corpus_data,
                        corpus_preds=corpus_pred,
                        weights=weights[test_id],
                        jacobian=jacobian,
                        decomposition_length=decomposition_size)
                        image_save_path = os.path.join("files","images",f"{d}_{cv}_{test_id}_{m}.png")
                        figure.savefig(image_save_path)
                writer.writerow([
                    time_stamp, 
                    corpus_size, 
                    test_size, 
                    decomposition_size, 
                    cv,
                    m.name, 
                    d.name,
                    latent_r2_score,
                    output_r2_score,
                    decompostion[test_id]["sample_id"],
                    decompostion[test_id]["target"],
                    decompostion[test_id]["decomposition"][0]["c_id"],
                    decompostion[test_id]["decomposition"][0]["c_weight"],
                    decompostion[test_id]["decomposition"][0]["c_target"],
                    corpus_ids,
                    corpus_weights,
                    corpus_targest,
                    image_save_path
                    ])

    return weights_all, latent_r2_scores, output_r2_scores, jacobians, decompostions

def run_ablation():
    """Run the different models for different combinations of corpus size, test size, decomposition size, seeding (cv) and test_id"""
    # testing 560 combinations
    print("Run ablation study.")
    print(f"Start time: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}")  # see https://stackoverflow.com/questions/415511/how-do-i-get-the-current-time-in-python
    start_time = time()
    corpus_size = [50, 100]
    test_size = [10, 50]
    decomposition_size = [5, 10, 50, 100]
    cv = [0,1]
    test_id = [0,1]
    for c in corpus_size:
        for t in test_size:
            for d in decomposition_size:
                if d > c:
                    continue
                for v in cv:
                    for id in test_id:
                        if id > (d-1):
                            continue
                        run_all_experiments(corpus_size=c, test_size=t, decomposition_size=d, cv=v, test_id=id, filename="ablation_results.csv")
    
    print(f"End time: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}") 
    print(f"The ablation study took {((time() - start_time) / 60):.0g} minutes.")


def run_original_experiment():
    """
    MNIST Approximation Quality Experiment as in paper and approximation_quality in original code (mnist.py). Using all 4 simplex models on MNIST dataset. 
    """

    models = [Model_Type.ORIGINAL, Model_Type.REIMPLEMENTED]
    datasets = list(Dataset)[:2] #TODO all 4
    decomposition_sizes = [3, 5]#, 10, 20, 50]
    cv_list = range(0,2)#TODO 10) # the results from the paper were obtained by taking all integer CV between 0 and 9
    explainer_names = set()
    results_df = pd.DataFrame(
        columns=[
            "explainer",
            "n_keep",
            "cv",
            "r2_latent",
            "r2_output",
        ]
    )
    
    # execute experiments for all decomposition sizes, cv's (different random seeds), first 3 simplex models ( original, compact original and reimplemented) on the original MNIST dataset
    for dec_s in decomposition_sizes:
        for cv in cv_list[:3]: #TODO remove debugging [0]
            w, l_r2, o_r2, jac, dec = run_all_experiments(corpus_size=10, test_size=5, decomposition_size=dec_s, cv=cv, test_id=0, filename="approximation_quality_results.csv", random_dataloader=True, no_ablation=True, plot_decomposition=False, datasets=datasets)  #TODO: with all datasets #TODO change back to 1000 100
            for d in datasets:
                for i, m in enumerate(models):
                    explainer_name = f"{m.name}_{d.name}"
                    results_df = pd.concat(
                        [
                            results_df,
                            pd.DataFrame.from_dict(
                                {
                                    "explainer": [explainer_name],
                                    "n_keep": [dec_s],
                                    "cv": [cv],
                                    "r2_latent": [l_r2[i]],
                                    "r2_output": [o_r2[i]],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                    explainer_names.add(explainer_name)

    metric_names = ["r2_latent", "r2_output"]
    styles = []
    colcors = [] #TODO
    #line_styles = {f"{explainer_names[0]}": "-", f"{explainer_names[1]}": ":"}#, f"{explainer_names[2]}": ":"}

    plt.rc("text", usetex=False)
    params = {"text.latex.preamble": r"\usepackage{amsmath}"}
    plt.rcParams.update(params)
    
    sns.set(font_scale=1.5)
    sns.set_style("white")
    sns.set_palette("colorblind")
    mean_df = results_df.groupby(["explainer", "n_keep"]).aggregate("mean", numeric_only=True).unstack(level=0)
    std_df = results_df.groupby(["explainer", "n_keep"]).aggregate(np.std).unstack(level=0)

    # my_xticks = decomposition_sizes TODO
    for m, metric_name in enumerate(metric_names):
        plt.figure(m + 1)
        for explainer_name in explainer_names:
            plt.plot(
                decomposition_sizes,
                mean_df[metric_name, explainer_name],
                #line_styles[explainer_name],   #TODO
                label=explainer_name,
            )
            plt.fill_between(
                decomposition_sizes,
                mean_df[metric_name, explainer_name] - std_df[metric_name, explainer_name],
                mean_df[metric_name, explainer_name] + std_df[metric_name, explainer_name],
                alpha=0.2,
            )

    save_path = os.path.join(SAVE_PATH, "original_experiment")
    timestamp = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    plt.figure(1)
    plt.xlabel(r"$decomposition size$")
    plt.ylabel(r"$R^2_{\mathcal{H}}$")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"r2_latent_{timestamp}.pdf"), bbox_inches="tight")
    plt.figure(2)
    plt.xlabel(r"$decomposition size$")
    plt.ylabel(r"$R^2_{\mathcal{Y}}$")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"r2_output{timestamp}.pdf"), bbox_inches="tight")
    #TODO adjust decomposition size in plot axis

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="These arguments determine which set of experiments is executed.")
    parser.add_argument(
        "-ablation",
        action="store_true",
        help="Run the tests for the ablation study and save it in \"/files/ablation_results.csv\"",
    )
    parser.add_argument(
        "-original",
        action="store_true",
        help="Run the original study from the paper and save it in \"/files/approximation_quality_results.csv\"",
    )
    parser.add_argument(
        "-all",
        action="store_true",
        help="Run experiments over all 4 datasets (4. is mnist with corpus_maker) and all 3 non-ablation simplex models and save it in \"/files/comparison_results.csv\"",
    )
    args = parser.parse_args()
    print(args)
    if args.ablation:
        run_ablation()
    elif args.original:
        run_original_experiment()
    elif args.all:
        run_all_experiments(no_ablation=True)
    else:
        run_original_experiment() #TODO remove
        parser.print_help()
        parser.exit()
    print("Done")

    # TODO: clean up code, obviously
