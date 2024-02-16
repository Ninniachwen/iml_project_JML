import inspect
import os
import sys

# access model in parent dir: https://stackoverflow.com/a/11158224/14934164
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from Original_Code.src.simplexai.utils.schedulers import ExponentialScheduler
from Original_Code.src.simplexai.explainers.simplex import Simplex

import torch

# settings use_case:
    # scheduler:
    #reg_factor_init=1.0; reg_factor_final=1000; n_epoch=20000  #epochs for simplex (epochs for mnist=10)
    #decompostion_size=100; test_id=0

    # settings toy_example:
    # scheduler:
    # reg_factor_init=0; x_final=100; n_epoch=20000
    # n_keep=decompostion_size=5; test_id=22

    # settings Yasmin:
    # reg_factor_init=0.1; x_final=100; n_epoch=10000
    # n_keep=decompostion_size=100; test_id=0

REG_FACTOR_INIT  = 1.0
REG_FACTOR_FINAL = 100
EPOCHS_ORIGINAL = 20000
EPOCHS_REIMPLEM = 20000 #TODO: 10000 or same as original?

def original_model(corpus_inputs, corpus_latents, test_inputs, test_latents, decompostion_size, test_id, classifier):  # jacobian
    # values take from "approximate_quality" in mnist.py
    reg_factor_scheduler = ExponentialScheduler(REG_FACTOR_INIT, x_final=REG_FACTOR_FINAL, n_epoch=EPOCHS_ORIGINAL) # test for step_factor=1.000691014168259 ?
    simplex = Simplex(corpus_examples=corpus_inputs,
            corpus_latent_reps=corpus_latents)    # test for corpus_size=100 & dim_latent=50
    simplex.fit(test_examples=test_inputs,
                test_latent_reps=test_latents,
                n_keep=decompostion_size,  # how many weights we want to keep in the end; keep all now to compare to own  model
                n_epoch=EPOCHS_ORIGINAL,
                reg_factor=REG_FACTOR_INIT,
                reg_factor_scheduler=reg_factor_scheduler)      # test for ?
    weights = simplex.weights   # test for shape=10,100 & requires_grad=False

    # see mnist.py approximation_quality
    latent_rep_approx = simplex.latent_approx()     # test for shape=10,50 & requires_grad=False
    
    # test unit -> if sum of top x weihgt adds up to at least ~90? or some other value a well trained original would have

    input_baseline = torch.zeros(corpus_inputs.shape)
    jacobian = simplex.jacobian_projection(test_id=test_id, model=classifier, input_baseline=input_baseline)

    return latent_rep_approx.detach(), weights.detach(), jacobian


def compact_original_model(corpus_inputs, corpus_latents, test_inputs, test_latents, decompostion_size, test_id, model):
    scheduler = ExponentialScheduler(x_init=0.1, x_final=REG_FACTOR_FINAL, n_epoch=EPOCHS_ORIGINAL)
    reg_factor = REG_FACTOR_INIT

    #simplex.fit TODO: make less similar to original...
    W_0 = torch.zeros((test_latents.shape[0], corpus_inputs.shape[0]), requires_grad=True) #test shape=10,100 & req gradient
    optimizer = torch.optim.Adam([W_0])
    for epoch in range(EPOCHS_ORIGINAL):
        #TODO: from simplex
        optimizer.zero_grad()
        weights = torch.nn.functional.softmax(W_0, dim=-1)      # test for shape=10,100 & requires_grad=False
        corpus_latent_reps = torch.einsum("ij,jk->ik", weights, corpus_latents)     # test for shape=10,50 & requires_grad=False
        error = ((corpus_latent_reps - test_latents) ** 2).sum()
        weights_sorted = torch.sort(weights)[0]
        regulator = (weights_sorted[:, : (corpus_inputs.shape[0] - decompostion_size)]).sum()
        loss = error + reg_factor * regulator
        loss.backward()
        optimizer.step()
        if (epoch + 1) % (EPOCHS_ORIGINAL / 5) == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{EPOCHS_ORIGINAL} ; Error: {error.item():.3g} ;"
                f" Regulator: {regulator.item():.3g} ; Reg Factor: {reg_factor:.3g}"
            )
        reg_factor = scheduler.step(reg_factor)
        #TODO: from simplex

    weights_softmax = torch.softmax(W_0, dim=-1).detach()
    # end of fit

    latent_rep_approx = weights_softmax @ corpus_latents    # reduction from  

    
    # weights_1_sample = weights_softmax[test_id].numpy() # only single test item out of batch
    # sort_id = np.argsort(weights_1_sample)[::-1]

    # corpus_decomposition = []
    # for i in sort_id:
    #     corpus_decomposition.append((weights_1_sample[i], corpus_inputs[i]))

    jacobian = []
    #TODO: detach jacobians too?
    return latent_rep_approx.detach(), weights_softmax.detach(), jacobian

    
def reimplemented_model(corpus_inputs, corpus_latents, test_inputs, test_latents, decompostion_size, test_id, model):
    size_test = test_latents.shape[0]
    size_corpus = corpus_inputs.shape[0]
    simplex = Simplex_Model(size_corpus, size_test)
    optimizer = torch.optim.Adam([simplex.weight]) # same optimizer as original
    for epoch in range(EPOCHS_REIMPLEM):
        optimizer.zero_grad()
        prediction = simplex(corpus_latents)
        loss = ((prediction - test_latents)** 2).sum() # same as original
        loss.backward()
        optimizer.step()
        # printing like in original
        if (epoch + 1) % (EPOCHS_ORIGINAL / 5) == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{EPOCHS_ORIGINAL} ; Error: {loss:.3g} ;"
            )
    

    weights = torch.nn.functional.softmax(simplex.weight, dim=1)

    # manually set the weights which should not be in the composition to 0
    weight_id_irrelevant = weights.argsort()[:,:(size_corpus - decompostion_size)]
    for i in range(size_test):
        for id in weight_id_irrelevant[i]:
            weights[i][id] = 0
    simplex.weight = weights

    # dont't use softmax after setting weights to 0 so the importance of weights is still the same
    latent_rep_approx = simplex(corpus_latents, softmax=False) 

    input_baseline = torch.zeros(corpus_inputs.shape)  # also as global parameter? 

    jacobian = simplex.get_jacobian(test_id, corpus_inputs, test_latents, input_baseline, model)

    return latent_rep_approx.detach(), weights.detach(), jacobian



class Simplex_Model(torch.nn.Module): #reimplemented
    # idea: do the training done in "fit"-Function of "simplex.py" in an more intuitive way
    # in original code, they cut down the inputs to only keep the most important corpus examples ("n_keep")while training
    # here, we train as normal and later set the not important weights to 0
    def __init__(self, size_corpus, size_test):
        super().__init__()
        self.weight = torch.zeros(size_test, size_corpus, requires_grad=True)
        # use basically a layer like torch.nn.Linear(size_corpus, size_test, bias=False)

    def forward(self, x, softmax=True):
        if softmax:
            weight = torch.nn.functional.softmax(self.weight,dim=1)
        else:  
            # with this (using the weights without softmax) during training, the model seems to overfit - 
            # the r2 values are better, but the weights are very close together
            weight = self.weight
        x = torch.matmul(weight, x)
        return x
    
    def get_jacobian(self, test_id, corpus_inputs, test_latents, input_baseline, classifier):
        # test_id: für welches test example die projections berechnet werden sollen
        latent_baseline = classifier.latent_representation(input_baseline)
        n_bins = 100 # standard in original
        # test_shift and test_shift_sqr like in simplex.py from original
        test_shift = test_latents[test_id] - latent_baseline
        test_shift_sqr = torch.sum(test_shift**2, dim=-1, keepdim=True)
        # the following is adapted from the integrated gradient exercise
        alphas = torch.linspace(0, 1, n_bins)
        all_gradients = 0
        for a in alphas:
            x_alpha = input_baseline + a * (corpus_inputs - input_baseline)
            x_alpha.requires_grad = True
            output = classifier.latent_representation(x_alpha)
            output.backward(test_shift/test_shift_sqr)
            all_gradients += x_alpha.grad
        jacobian = (corpus_inputs - input_baseline) * (all_gradients / n_bins)
        return jacobian