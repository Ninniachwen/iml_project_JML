import inspect
import os
import sys
import torch
from original_code.src.simplexai.models.image_recognition import MnistClassifier

sys.path.insert(0, "")
from original_code.src.simplexai.utils.schedulers import ExponentialScheduler
from original_code.src.simplexai.explainers.simplex import Simplex


# values take from "approximate_quality" in mnist.py
REG_FACTOR_INIT  = 0.1
REG_FACTOR_FINAL = 100
EPOCHS = 10000

def original_model(corpus_inputs, corpus_latents, test_inputs, test_latents, decompostion_size:int, test_id:int, classifier, input_baseline:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The original simplex model: it trains the model and creates jacobians for the given test_id.

    Args:
        corpus_inputs (torch.Tensor): Feature vector of Corpus examples
        corpus_latents (torch.Tensor): Latent representations of Corpus examples, run through according blackbox model
        test_data (torch.Tensor): Feature vector of Test examples (not used, but given as argument to keep function calls the same)
        test_latents (torch.Tensor): Latent representations of Test examples, run through according model
        decompostion_size (int): with how many corpus examples a test example should be explained
        test_id (int): for which Test example the jacobians should be printed
        classifier (_type_): the original blackbox model (only needed for jacobians)
        input_baseline (torch.Tensor): baseline for jacobian projections

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            1. latent_rep_approx: the latent representations learned by the simplex model; used for r2 score.
            2. weights_softmax: the (softmaxed) weights of the learned simplex model
            3. jacobian: the jacobians for the given test id
    """
    reg_factor_scheduler = ExponentialScheduler(REG_FACTOR_INIT, x_final=REG_FACTOR_FINAL, n_epoch=EPOCHS)
    simplex = Simplex(corpus_examples=corpus_inputs,
            corpus_latent_reps=corpus_latents)
    simplex.fit(test_examples=test_inputs,
                test_latent_reps=test_latents,
                n_keep=decompostion_size,  # how many weights we want to keep in the end; keep all now to compare to own  model
                n_epoch=EPOCHS,
                reg_factor=REG_FACTOR_INIT,
                reg_factor_scheduler=reg_factor_scheduler)
    weights = simplex.weights

    # see mnist.py approximation_quality
    latent_rep_approx = simplex.latent_approx()
    
    # test unit -> if sum of top x weihgt adds up to at least ~90? or some other value a well trained original would have

    jacobian = simplex.jacobian_projection(test_id=test_id, model=classifier, input_baseline=input_baseline)

    return latent_rep_approx.detach(), weights.detach(), jacobian.detach()


def compact_original_model(corpus_inputs:torch.Tensor, corpus_latents:torch.Tensor, test_data:torch.Tensor, test_latents:torch.Tensor, decompostion_size:int, test_id:int, classifier, input_baseline:torch.Tensor, softmax=True, regularisation=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    In this method the original Simplex model (from simplex.py) is collected into a single method (and one helper method below). This simplifies the ablation study. Othewise it works just like Simplex. This method trains the model and creates jacobians for given test_id.     

    Args:
        corpus_inputs (torch.Tensor): Feature vector of Corpus examples
        corpus_latents (torch.Tensor): Latent representations of Corpus examples, run through according blackbox model
        test_data (torch.Tensor): Feature vector of Test examples (not used, but given as argument to keep function calls the same)
        test_latents (torch.Tensor): Latent representations of Test examples, run through according model
        decompostion_size (int): with how many corpus examples a test example should be explained
        test_id (int): for which Test example the jacobians should be printed
        classifier (_type_): the original blackbox model (only needed for jacobians)
        input_baseline (torch.Tensor): baseline for jacobian projections
        softmax (bool, optional): True means the softmax layer should be used; used for ablation study. Defaults to True.
        regularisation (bool, optional): True means the regularization factor is used; used for ablation study. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            1. latent_rep_approx: the latent representations learned by the simplex model; used for r2 score.
            2. weights_softmax: the (softmaxed) weights of the learned simplex model
            3. jacobian: the jacobians for the given test id
    """
    scheduler = ExponentialScheduler(x_init=0.1, x_final=REG_FACTOR_FINAL, n_epoch=EPOCHS)
    if regularisation:
        reg_factor = REG_FACTOR_INIT
    else:
        reg_factor = 0

    # simplex.fit
    W_0 = torch.zeros((test_latents.shape[0], corpus_inputs.shape[0]), requires_grad=True) 
    optimizer = torch.optim.Adam([W_0])
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        if softmax:
            weights = torch.nn.functional.softmax(W_0, dim=-1)
        else:
            weights = W_0
        corpus_latent_reps = torch.einsum("ij,jk->ik", weights, corpus_latents)
        error = ((corpus_latent_reps - test_latents) ** 2).sum()
        weights_sorted = torch.sort(weights)[0]
        regulator = (weights_sorted[:, : (corpus_inputs.shape[0] - decompostion_size)]).sum()
        loss = error + reg_factor * regulator
        loss.backward()
        optimizer.step()
        if (epoch + 1) % (EPOCHS / 5) == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{EPOCHS} ; Error: {error.item():.3g} ;"
                f" Regulator: {regulator.item():.3g} ; Reg Factor: {reg_factor:.3g}"
            )
        reg_factor = scheduler.step(reg_factor)
    
    weights_softmax = torch.softmax(W_0, dim=-1).detach()
    # end of fit

    if softmax:
        latent_rep_approx = weights_softmax @ corpus_latents    # reduction from  
    else:
        latent_rep_approx = weights @ corpus_latents 

    jacobian = compact_jacobian_projections(corpus_inputs, latent_rep_approx, test_id, classifier, input_baseline)
    
    return latent_rep_approx.detach(), weights_softmax.detach(), jacobian.detach()


def compact_jacobian_projections(corpus_inputs:torch.Tensor, corpus_latents:torch.Tensor, test_id:int, model:MnistClassifier, input_baseline:torch.Tensor, n_bins=100) -> torch.Tensor:  
    """
    Compute the Jacobian Projection for the test examples
    :param test_id: batch index of the test example
    :param model: the black-box model for which the Jacobians are computed
    :param input_baseline: the baseline input features
    :param n_bins: number of bins involved in the Riemann sum approximation for the integral
    :return:
    """
    input_shift = corpus_inputs - input_baseline
    corpus_data = corpus_inputs.clone().requires_grad_()
    latent_shift = corpus_latents[
        test_id : test_id + 1
    ] - model.latent_representation(input_baseline)
    latent_shift_sqrdnorm = torch.sum(latent_shift**2, dim=-1, keepdim=True)
    input_grad = torch.zeros(corpus_data.shape, device=corpus_data.device)
    for n in range(1, n_bins + 1):
        t = n / n_bins
        input = input_baseline + t * (corpus_data - input_baseline)
        latent_reps = model.latent_representation(input)
        latent_reps.backward(gradient=latent_shift / latent_shift_sqrdnorm)
        input_grad += corpus_data.grad
        corpus_data.grad.data.zero_()
    jacobian_projections = input_shift * input_grad / (n_bins)
    return jacobian_projections

    
def reimplemented_model(corpus_inputs:torch.Tensor, corpus_latents:torch.Tensor, test_data:torch.Tensor, test_latents:torch.Tensor, decompostion_size:int, test_id:int, classifier, input_baseline:torch.Tensor, mode="softmax", weight_init_zero=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Our reimplemented simplex model: it trains the model and creates jacobians for given test_id.

    Args:
        corpus_inputs (torch.Tensor): Feature vector of Corpus examples
        corpus_latents (torch.Tensor): Latent representations of Corpus examples, run through according blackbox model
        test_data (torch.Tensor): Feature vector of Test examples (not used, but given as argument to keep function calls the same)
        test_latents (torch.Tensor): Latent representations of Test examples, run through according model
        decompostion_size (int): with how many corpus examples a test example should be explained
        test_id (int): for which Test example the jacobians should be printed
        classifier (_type_): the original blackbox model (only needed for jacobians)
        input_baseline (torch.Tensor): baseline for jacobian projections
        softmax (bool, optional): True means the softmax layer should be used; used for ablation study. Defaults to True.
        mode (str, optional): "softmax", "normalize" or "nothing"; diffenrent modes for training the simplex model; used for ablation study. Defaults to "softmax".
        weight_init_zero (bool, optional): True means we initialize the weights of the simple model with zeros (therwise random); used for ablation study. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            1. latent_rep_approx: the latent representations learned by the simplex model; used for r2 score.
            2. weights: the weights of the learned simplex model
            3. jacobian: the jacobians for the given test id
    """
    size_test = test_latents.shape[0]
    size_corpus = corpus_inputs.shape[0]
    simplex = Simplex_Model(size_corpus, size_test, weight_init_zero=weight_init_zero)
    optimizer = torch.optim.Adam([simplex.weight]) # same optimizer as original
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        prediction = simplex(corpus_latents, mode=mode)
        loss = ((prediction - test_latents)** 2).sum() # same loss as original
        loss.backward()
        optimizer.step()
        # printing like in original
        if (epoch + 1) % (EPOCHS / 5) == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{EPOCHS} ; Error: {loss:.3g} ;"
            )
    
    weights = simplex.weight.clone()

    # TODO: nochmal checken: soll weights hier gesoftmaxt werden bevor man sie auf 0 setzt? ausprobieren
    # TODO normalize funltioniert nicht wie es sein solle - ändere und mach händisch weights / weights.sum(dim=1) oder so? auch bei forward?
    if mode == "softmax":
        weights = torch.nn.functional.softmax(weights, dim=1)
    elif mode == "normalize":
        weights = torch.nn.functional.normalize(abs(weights),dim=1)

    # manually set the weights which should not be in the decomposition to 0
    weight_id_irrelevant = weights.argsort()[:,:(size_corpus - decompostion_size)]
    for i in range(size_test):
        for id in weight_id_irrelevant[i]:
            weights[i][id] = 0

    simplex.weight = weights



   # weights_softmax = torch.nn.functional.softmax(simplex.weight, dim=1)

    latent_rep_approx = simplex(corpus_latents, mode="nothing") # because we softmaxed /normalized if already before

    jacobian = simplex.get_jacobian(test_id, corpus_inputs, test_latents, input_baseline, classifier)

    #if mode == "softmax":  # 
    #    weights = weights_softmax

    return latent_rep_approx.detach(), weights.detach(), jacobian


class Simplex_Model(torch.nn.Module):
    """Our reimplemented model. """
    # idea: do the training done in "fit"-Function of "simplex.py" in an more intuitive way
    # in original code, they used regularization to prioritize the X most important corpus examples ("n_keep") while training
    # here, we train as normal and later set the not important weights to 0
    def __init__(self, size_corpus:int, size_test:int, weight_init_zero=True)->None:
        super().__init__()
        if weight_init_zero:
            # same as original
            self.weight = torch.zeros(size_test, size_corpus, requires_grad=True)
        else:
            self.weight = torch.randn(size_test, size_corpus, requires_grad=True)
        # use basically a layer like torch.nn.Linear(size_corpus, size_test, bias=False)

    def forward(self, x:torch.Tensor, mode="softmax") ->torch.Tensor:
        """ Forward pass. Use diffenrent modes (softmax, normalize, nothing) for ablation study."""
        if mode =="softmax":
            weight = torch.nn.functional.softmax(self.weight,dim=1)
        elif mode == "normalize":
            # abs() because we only want positive weights
            weight = torch.nn.functional.normalize(abs(self.weight),dim=1)
        elif mode=="nothing":  
            # with this (using the weights without softmax) during training, the model seems to overfit - 
            # the r2 values are better (with decomposition = 100), but the weights are very close together
            weight = self.weight
        else:
            raise Exception(f"'{mode}' is no valid input for mode!")
        x = torch.matmul(weight, x)
        return x
    
    def get_jacobian(self, test_id:int, corpus_inputs:torch.Tensor, test_latents:torch.Tensor, input_baseline:torch.Tensor, classifier) -> torch.Tensor:
        """Create jacobian projections for given test id. The jacobian projection was introduced
        by the authors of the paper and is basically a generalization of the integrated gradients."""
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