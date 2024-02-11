
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

def original_model(n_epoch, corpus_inputs, corpus_latents, test_inputs, test_latents, decompostion_size):    #, test_id, classifier):  # jacobian
    # values take from "approximate_quality" in mnist.py
    reg_factor_scheduler = ExponentialScheduler(REG_FACTOR_INIT, x_final=REG_FACTOR_FINAL, n_epoch=n_epoch) # test for step_factor=1.000691014168259 ?
    simplex = Simplex(corpus_examples=corpus_inputs,
            corpus_latent_reps=corpus_latents)    # test for corpus_size=100 & dim_latent=50
    simplex.fit(test_examples=test_inputs,
                test_latent_reps=test_latents,
                n_keep=decompostion_size,  # how many weights we want to keep in the end; keep all now to compare to own  model
                n_epoch=n_epoch,
                reg_factor=REG_FACTOR_INIT,
                reg_factor_scheduler=reg_factor_scheduler)      # test for ?
    #weights = simplex.weights   # test for shape=10,100 & requires_grad=False
    
    # see mnist.py approximation_quality
    latent_rep_approx = simplex.latent_approx()     # test for shape=10,50 & requires_grad=False
    
    # test unit -> if sum of top x weihgt adds up to at least ~90? or some other value a well trained original would have

    #input_baseline = torch.zeros(corpus_inputs.shape)
    #jacobian = simplex.jacobian_projection(test_id=test_id, model=classifier, input_baseline=input_baseline)

    return latent_rep_approx


def compact_original_model(n_epoch, corpus_inputs, corpus_latents, test_inputs, test_latents, decompostion_size):
    scheduler = ExponentialScheduler(x_init=0.1, x_final=REG_FACTOR_FINAL, n_epoch=n_epoch)
    reg_factor = REG_FACTOR_INIT

    #simplex.fit TODO: make less similar to original...
    W_0 = torch.zeros((test_latents.shape[0], corpus_inputs.shape[0]), requires_grad=True) #test shape=10,100 & req gradient
    optimizer = torch.optim.Adam([W_0])
    for epoch in range(n_epoch):
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
        if (epoch + 1) % (n_epoch / 5) == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{n_epoch} ; Error: {error.item():.3g} ;"
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

    # jacobian = []

    return latent_rep_approx

    
def reimplemented_model(n_epoch, corpus_inputs, corpus_latents, test_inputs, test_latents, decompostion_size):
    size_test = test_latents.shape[0]
    size_corpus = corpus_inputs.shape[0]
    model = Simplex_Model(size_corpus, size_test)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001, lr=0.001) # same optimizer as original #TODO: original uses no weight decay, learning rate is the same, because default
    # play with parameters to not overfit model
    for epoch in range(1000):
        optimizer.zero_grad()
        prediction = model(corpus_latents)
        loss = ((prediction - test_latents)** 2).sum() # same as original
        loss.backward()
        optimizer.step()
        
        #print(loss)
    weights = model.layer1.weight # shape: 10,100,1 (size_test,size_corpus,1)
    weights = weights.reshape([size_test,size_corpus])
    weights = torch.nn.functional.softmax(weights, dim=1)

    latent_rep_approx = model(corpus_latents)

    # jacobian does not work yet
    # jacobian = []

    return latent_rep_approx



class Simplex_Model(torch.nn.Module):
    # idea: do the training done in "fit"-Function of "simplex.py" in an more intuitive way
    # in original code, they cut down the inputs to only keep the most important corpus examples ("n_keep") -> lets ignore that for now
    def __init__(self, size_corpus, size_test):
        super().__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=size_corpus, out_channels=size_test,kernel_size=1, bias=False)
        self.layer1.weight.data.fill_(0) # initialize weights with 0s
    def forward(self, x):
        x = self.layer1(x)
        return x