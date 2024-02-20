import csv
from pathlib import Path
from captum.attr._utils.visualization import visualize_image_attr
import enum
import inspect
import numpy as np
import os
import torch
import sys
import argparse

#TODO: check if requirements file is sufficient


# access model in parent dir: https://stackoverflow.com/a/11158224/14934164
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.evaluation as e
import src.simplex_versions as s
import src.classifier_versions as c

RANDOM_SEED=42

class Model_Type(enum.Enum):
    ORIGINAL = 1  # use original model grom github repository
    ORIGINAL_COMPACT = 2  # use original code which was rewritten to fit in one function
    REIMPLEMENTED = 3  # use reimplemented model
    R_NO_SOFTMAX = 4  # use reimplemented model but without softmax layer during training (for ablation)
    R_NORMALIZE = 5  # use reimplemented model but using a normalization layer during training (for ablation)
    R_INIT_WEIGHTS_RANDOM = 6 # use reimplemented model but use set initial weights with random values (for ablation)
    R_NO_SOFTMAX_IWR = 7  # like R_NO_SOFTMAX but use set initial weights with random values (for ablation)
    R_NORMALIZE_IWR = 8  # like R_NORMALIZE but use set initial weights with random values (for ablation)
    O_C_NO_SOFTMAX = 9  # use compact original (ORIGINAL_COMPACT) but without softmax layer during training (for ablation)
    O_C_NO_REGULIZATION = 10  # use condensed original (ORIGINAL_COMPACT) but use no regularization during training (for ablation)


class Dataset(enum.Enum):
    MNIST = 1
    #CaN = 2
    #Heart = 3


# code mostly from the toy example from their github page right now,  also referencing use_case.py

# when using another dataset, we also have to replace load_minst (with our custom loader) and MnistClassifier (maybe with another pretrained model)

# for this to work right now you have to follow the install instructions of the original repo


def do_simplex(model_type=Model_Type.ORIGINAL, dataset=Dataset.MNIST, cv=0, decomposition_size=100, corpus_size=100, test_size=10, test_id=0, print_jacobians=False, r_2_scores=True, decompose=True, random_dataloader=True):
    """ #TODO: aktualisieren
    Decide if we want to train our Simplex model the original or the new implemented way

    Parameters:
        model_type (enum|int): ORIGINAL = 1, REIMPLEMENTED = 2, NO_REGULARIZATION = 3, TORCH_CONV = 4
        corpus_inputs: Feature vector of Corpus examples; shape ?
        corpus_latents: Latent representations of Corpus examples, run through according model; shape ?
        test_inputs: Feature vector of Test examples; shape ?
        test_latents: Latent representations of Test examples, run through according model; shape ?
        classifier: the classifier used (only for output score)
        cv: for seeding, see original code (cross validation parameter)
    
    Returns:
        weights: Weights of the trained Model
        latent_r2_score: r2 score of the learned representation
        output_r2_score: r2 score of output (using original model) based on learned representation
        #-no-# jacobian: the jacobian projection of the first element

        The score-functions are taken directly from mnist.py of the original code
    """

    assert corpus_size >= decomposition_size, "decomposition size can't be larger than corpus"
    assert test_size > test_id, "test_id can't be larger than test_size"

    torch.random.manual_seed(RANDOM_SEED + cv)
    torch.backends.cudnn.deterministic = True # https://www.typeerror.org/docs/pytorch/generated/torch.nn.conv1d

    # must return classifier, (corpus_data, corpus_latents, corpus_target), (test_data, test_targets, test_latents)
    if dataset is Dataset.MNIST:
        classifier, corpus, test_set = c.train_or_load_mnist(RANDOM_SEED, cv, corpus_size=corpus_size, test_size=test_size, random_dataloader=random_dataloader)
        #TODO: maybe keep as triples and hand triples to models
        corpus_data, corpus_target, corpus_latents = corpus
        test_data, test_targets, test_latents = test_set
    
    else:
        raise Exception(f"'{dataset}' is no valid input for dataset")
        #TODO: test for this exception
    
    # must return latent_rep_approx, weights, jacobian
    if model_type is Model_Type.ORIGINAL:
        print(f"Starting on cv {cv} with the original model!")
        latent_rep_approx, weights, jacobian = s.original_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier)

    elif model_type is Model_Type.ORIGINAL_COMPACT:
        print(f"Starting on cv {cv} with the compact original model!")
        latent_rep_approx, weights, jacobian = s.compact_original_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier)
    
    elif model_type is Model_Type.O_C_NO_SOFTMAX:
        print(f"Starting on cv {cv} with the compact original model but without using softmax layer while training!")
        latent_rep_approx, weights, jacobian = s.compact_original_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, softmax=False)
        
    elif model_type is Model_Type.O_C_NO_REGULIZATION:
        print(f"Starting on cv {cv} with the compact original model but without using regularization while training!")
        latent_rep_approx, weights, jacobian = s.compact_original_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, regularisation=False)
        
    elif model_type is Model_Type.REIMPLEMENTED:
        print(f"Starting on cv {cv} with our own reimplemented model!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier)
    
    elif model_type is Model_Type.R_NORMALIZE:
        print(f"Starting on cv {cv} with the compact original model but using normalize instead of softmax layer!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, mode="normalize")
        
    elif model_type is Model_Type.R_NO_SOFTMAX:
        print(f"Starting on cv {cv} with our own reimplemented model but without using softmax layer!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, mode="nothing")

    elif model_type is Model_Type.R_INIT_WEIGHTS_RANDOM:
        print(f"Starting on cv {cv} with our own reimplemented model but initializing weights at random!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, weight_init_zero=False)
    
    elif model_type is Model_Type.R_NO_SOFTMAX_IWR:
        print(f"Starting on cv {cv} with the compact original model but using normalize instead of softmax layer and initializing weights at random!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, mode="normalize", weight_init_zero=False)
        
    elif model_type is Model_Type.R_NORMALIZE_IWR:
        print(f"Starting on cv {cv} with our own reimplemented model but without using softmax layer and initializing weights at random!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decomposition_size, test_id, classifier, mode="nothing", weight_init_zero=False)
        

    else:
        raise Exception(f"'{model_type}' is no valid input for model_type")
        #TODO: test for this exception
    
    latent_rep_true = test_latents  # test for shape=10,50 & requires_grad=False
    
    latent_r2_score = None
    output_r2_score = None
    if r_2_scores:
        output_r2_score, latent_r2_score = e.r_2_scores(classifier, latent_rep_approx, latent_rep_true, weights, test_id, test_data, corpus_data)
    
    decompostions = None
    if decompose:
        #TODO: use latents instead of weights?
        decompostions = e.create_decompositions(test_data, test_targets, corpus_data, corpus_target, decomposition_size, weights)

    if print_jacobians:

        #most_imp_id = decompostions[test_id]["decomposition"][0]["c_id"]
        #saliency = jacobian[most_imp_id].numpy().transpose((1, 2, 0))

        # [Jasmin:] The following works for printing the jacobians. We may want to print them 
        # in a different file / with some global parameters / using the decomposition
        most_important_example = weights[test_id].argmax()
        image = corpus_data[most_important_example].numpy().transpose((1, 2, 0))  # transpose see use_case.py
        saliency = jacobian[most_important_example].numpy().transpose((1, 2, 0))  # transpose see use_case.py
        # the following code, see use_case.py
        fig3, axis = visualize_image_attr(
                saliency,
                image,
                method="blended_heat_map",
                sign="all",
                title="Jacobian of most important corpus example",
                use_pyplot=True,
            )
        most_imp_id = decompostions[test_id]["decomposition"][0]["c_id"]
        saliency = jacobian[most_imp_id].numpy().transpose((1, 2, 0))
    
    return weights, latent_r2_score, output_r2_score, jacobian, decompostions


def run_all_experiments(corpus_size=100, test_size=10, decomposition_size=3, cv=0, test_id=0, filename="comparison_results.csv", random_dataloader=False): 
    #TODO: remove duplicate name test_id. (ok in csv, should be called different in rest of code (this function and others). (sample_id?))

    print(f"   starting test runs with parameters:\n   corpus_size: {corpus_size}, test_size: {test_size}, decomposition_size: {decomposition_size}, cv: {cv}, test_id: {test_id}")
              
    weights_all = []
    latent_r2_scores = []
    output_r2_scores = []
    jacobians = []
    decompostions = []
    
    running_int = 0
    file_path = os.path.join(parentdir, "files" , filename)
    file = Path(file_path)
    mode = "a" if file.is_file() else "w"   # append if file exists, assuming either both files exist, or none
    with open(file_path, mode) as f1:
    
        writer=csv.writer(f1, delimiter=";",lineterminator="\n",)

        if mode == "w":     # add header if file new
            writer.writerow([
                        "experiment_id",
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
                        #TODO: generate image of decomposition, store & link
                        ])
            
        for d in Dataset:
            for m in Model_Type:
                weights, latent_r2_score, output_r2_score, jacobian, decompostion = do_simplex(
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

                writer.writerow([
                    test_id, 
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
                    #TODO: generate image of decomposition, store & link
                    ])
                running_int += 1

    return weights_all, latent_r2_scores, output_r2_scores, jacobians, decompostions

def run_ablation():
    """Run the different models for different combinations of corpus size, test size, decomposition size, seeding (cv) and test_id"""
    # testing 540 combinations
    print("Run ablation study.")
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
                            pass
                        run_all_experiments(corpus_size=c, test_size=t, decomposition_size=d, cv=v, test_id=id, filename="ablation_results.csv")


def run_original_experiment():
    # MNIST Approximation Quality Experiment
    # as in approximation_quality in original simplex
    decomposition_size = [3, 5, 10, 20, 50]
    cv = range(0,10)
    for d in decomposition_size:
        for v in cv:    # the results from the paper were obtained by taking all integer CV between 0 and 9
            run_all_experiments(corpus_size=1000, test_size=100, decomposition_size=d, cv=v, test_id=0, filename="approximation_quality_results.csv", random_dataloader=True)
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO!")
    parser.add_argument(
        "-ablation",
        action="store_true",
        help="Run the tests for the ablation study and save it \"/files/ablation_results.csv\"",
    )
    args = parser.parse_args()
    print(args)
    if args.ablation:
        run_ablation()
    else:
        parser.print_help()
        parser.exit()

    print("Done")
    #TODO: join test_set into corpus and see what happens
   
    # TODO: more seeding, values are slightly of each time of run (original and own)
    # TODO  sanity check: is our model "too good"? should be do something even easier than 1 conv layer? 
    # TODO: deal with different "to_keep" values of weights, alter funciton -> check exactly what is done with them in fit-method of simplex.py
    # TODO: introduce jacobian projection: but how to test ??
    # TODO: generalize simplex_model to not only work für mnist--> done
    # TODO: introduce plotting; later: maybe add own model score to their plot (with the nearest neighbors?)
    # TODO: do own mnist classifier and train it in our own way? that will probably be some work. is it necessary?
    # TODO: clean up code, obviously
    # TODO: plot the according pictures to get a better understanding

    # next steps (after the one above):
    # other dataset, ablation etc.