import inspect
import sys
import os
# access model in parent dir: https://stackoverflow.com/a/11158224/14934164
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)   
from Original_Code.src.simplexai.models.image_recognition import MnistClassifier
from Original_Code.src.simplexai.experiments import mnist

import models as m


import torch
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

#SAVE_PATH="../experiments/results/mnist/quality/" #Jasmin
SAVE_PATH=os.path.join(parentdir, "files")
RANDOM_SEED=42

class Model_Type(Enum):
    ORIGINAL = 1
    ORIGINAL_COMPACT = 2
    REIMPLEMENTED = 3
    R_NO_SOFTMAX = 4
    O_C_NO_SOFTMAX = 5
    O_C_NO_REGULIZATION = 6

class dataset(Enum):
    MNIST = 1
    OTHER = 2

# code mostly from the toy example from their github page right now,  also referencing use_case.py

# when using another dataset, we also have to replace load_minst (with our custom loader) and MnistClassifier (maybe with another pretrained model)

# for this to work right now you have to follow the install instructions of the original repo

# directly from original code visualization/images.py
def plot_mnist(data, title: str = "") -> plt.Figure:
    fig = plt.figure()
    plt.imshow(data, cmap="gray", interpolation="none")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return fig


def do_simplex(model_type, data, cv, n_epoch, decompostion_size, test_id):
    """
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
    torch.random.manual_seed(RANDOM_SEED + cv)
    torch.backends.cudnn.deterministic = True # https://www.typeerror.org/docs/pytorch/generated/torch.nn.conv1d

    if data is dataset.MNIST:
        model_reg_factor=0.1
        if not os.path.isfile(os.path.join(SAVE_PATH,f"model_cv{cv}.pth")):
            mnist.train_model(
                    device="cpu",
                    random_seed=RANDOM_SEED,
                    cv=cv,
                    save_path=SAVE_PATH,
                    model_reg_factor=model_reg_factor,
                )
        
        model = MnistClassifier()
        model.load_state_dict(torch.load(os.path.join(SAVE_PATH,f"model_cv{cv}.pth")))
        # model.to(device) porbably not necessary
        model.eval()       

        # data loader from approximate_quality
        corpus_loader = mnist.load_mnist(100, train=True, shuffle=False)
        test_loader = mnist.load_mnist(10, train=False, shuffle=False)
        batch_id_test , (test_data, test_targets) = next(enumerate(test_loader))
        batch_id_corpus, (corpus_data, corpus_target) = next(enumerate(corpus_loader))
        corpus_data = corpus_data.detach() #.to(device).detach()   #TODO: remove to device everywhere for consistency
        test_data = test_data.detach() #.to(device).detach()
        corpus_latents = model.latent_representation(corpus_data).detach()
        test_latents = model.latent_representation(test_data).detach()


    if model_type is Model_Type.ORIGINAL:
        print(f"Starting on cv {cv} with the original model!")
        latent_rep_approx, weights = m.original_model(n_epoch, corpus_data, corpus_latents, test_data, test_latents, decompostion_size)


    if model_type is Model_Type.ORIGINAL_COMPACT:
        print(f"Starting on cv {cv} with the compact original model!")
        latent_rep_approx, weights = m.compact_original_model(n_epoch, corpus_data, corpus_latents, test_data, test_latents, decompostion_size)
        
    if model_type is Model_Type.REIMPLEMENTED:
        print(f"Starting on cv {cv} with our own reimplemented model!")
        latent_rep_approx, weights = m.reimplemented_model(n_epoch, corpus_data, corpus_latents, test_data, test_latents, decompostion_size)

    latent_rep_true = test_latents  # test for shape=10,50 & requires_grad=False
    
    #TODO: remove tests to seperate file
    #TODO: create test for decomposition

    output_approx = model.latent_to_presoftmax(latent_rep_approx).detach()
    output_true = model.latent_to_presoftmax(latent_rep_true).detach()
    output_r2_score = sklearn.metrics.r2_score(
        output_true, output_approx
    )
    latent_r2_score = sklearn.metrics.r2_score(
                latent_rep_true, latent_rep_approx.detach().numpy()
            )
    
    most_important_example = weights[test_id].argmax()  # for test example 0
    fig1 = plot_mnist(test_data[test_id][0], f"Test example {test_id}")
    fig2 = plot_mnist(corpus_data[most_important_example][0], f"M.i. attribution to example 0 (corpus id {most_important_example})")
    fig1.show()
    fig2.show()

    print(f"Biggest contributor to test example 0: {weights[0].argmax()} with weight {weights[0].max()}")
    print(weights[0])
    # if shuffle=False in corpus- and test-loader, the get the same results (but ir's logically also always the same for all cvs)!
    return weights, latent_r2_score, output_r2_score # , jacobian

#  auswertungen:
# 1. r2 repr (jasmin)
# 2. welche bilder sind die top x erklärungen, zahl, id, weights
# doch nicht: # 2. r2 score zahl (jasmin)


def do_mnist_experiment(cv=0):
    # the following are standard values which are used in mnist.py to train mnistClassifier

    # settings use_case:
    # scheduler:
    reg_factor_init=1.0; reg_factor_final=1000; n_epoch=20000  #epochs for simplex (epochs for mnist=10)
    decompostion_size=100; test_id=0

    # settings toy_example:
    # scheduler:
    # reg_factor_init=0; x_final=100; n_epoch=20000
    # n_keep=decompostion_size=5; test_id=22

    # settings Yasmin:
    # reg_factor_init=0.1; x_final=100; n_epoch=10000
    # n_keep=decompostion_size=100; test_id=0
  
    
    # original_model, original_score_latent, original_score_output, original_jacobian
    original = do_simplex(Model_Type.ORIGINAL, dataset.MNIST, cv, n_epoch, decompostion_size, test_id)

    # reimplemented_model, reimplemented_score_latent, reimplemented_score_output, reimplemented_jacobian
    reimplemented = do_simplex(Model_Type.ORIGINAL_COMPACT, dataset.MNIST, cv, n_epoch, decompostion_size, test_id)

    #own_model, own_score_latent, own_score_output, own_jacobian
    jasmin = do_simplex(Model_Type.REIMPLEMENTED, dataset.MNIST, cv, n_epoch, decompostion_size, test_id)

    print(original[1], original[2]) # for cv=1 : 0.9178502448017772 0.9458505321920692
    print(reimplemented[1], reimplemented[2]) # for cv=1 : 0.9178502448017772 0.9458505321920692
    print(jasmin[1], jasmin[2]) # for cv=1 :0.9999924820980085 0.999999255618875

    return original[1], original[2], reimplemented[1], reimplemented[2], jasmin[1], jasmin[2]

    # the original model seems to get worse the more times i run it ....


def run_multiple_experiments():
    original_score_latents = []
    original_score_outputs = []
    own_score_latents = []
    own_score_outputs = []
    #for i in range(10):
    for i in range(1):
        original_score_latent, original_score_output, own_score_latent, own_score_output = do_mnist_experiment(i)
        original_score_latents.append(original_score_latent)
        original_score_outputs.append(original_score_output)
        own_score_latents.append(own_score_latent)
        own_score_outputs.append(own_score_output)
    
    print(f"Original score latents: {original_score_latents}; mean: {np.mean(original_score_latents)}")
    print(f"Original score outputs: {original_score_outputs}; mean: {np.mean(original_score_outputs)}")
    print(f"Own score latents: {own_score_latents}; mean: {np.mean(own_score_latents)}")
    print(f"Own score outputs: {own_score_outputs}; mean: {np.mean(own_score_outputs)}")

    # if shuffle=False in corpus- and test-loader, the original values get better (and more consistent)!

if __name__ == "__main__":

    # training mnist, from minst.py
    #run_multiple_experiments()
    do_mnist_experiment()
   
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