from Original_Code.src.simplexai.explainers.simplex import Simplex
from Original_Code.src.simplexai.models.image_recognition import MnistClassifier
from Original_Code.src.simplexai.experiments.mnist import load_mnist
from Original_Code.src.simplexai.experiments.mnist import train_model
from Original_Code.src.simplexai.utils.schedulers import ExponentialScheduler


import torch
import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Model_Type(Enum):
    ORIGINAL = 1
    REIMPLEMENTED = 2
    NO_REGULARIZATION = 3
    TORCH_CONV = 4


# just writing some code here first to get the functionalities of Simplex in one file
# code mostly from the toy example from their github page right now,  also referencing use_case.py

# later, we want to reimplement Simplex so we don't have to import it
# when using another dataset, we also have to replace load_minst (with our custom loader) and MnistClassifier (maybe with another pretrained model)

# for this to work right now you have to follow the install instructions of the original repo

# directly from original code visualization/images.py
def plot_mnist(data, title: str = "") -> plt.Figure:
    fig = plt.figure()
    plt.imshow(data, cmap="gray", interpolation="none")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return plt


def do_simplex(model_type, corpus_inputs, corpus_latents, test_inputs, test_latents, classifier, cv, reg_factor_init, reg_factor_final, n_epoch, decompostion_size, test_id):
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
        jacobian: the jacobian projection of the first element
        TODO: representer_output_r2_score: how well the simplex model is as a classifier 

        The score-functions are taken directly from mnist.py of the original code
    """
    random_seed=42

    size_corpus = corpus_latents.shape[0]
    size_test = test_latents.shape[0]

    input_baseline = torch.zeros(corpus_inputs.shape)

    torch.random.manual_seed(random_seed + cv)
    torch.backends.cudnn.deterministic = True # https://www.typeerror.org/docs/pytorch/generated/torch.nn.conv1d


    if model_type is Model_Type.ORIGINAL:
        print(f"Starting on cv {cv} with the original model!")
        # values take from "approximate_quality" in mnist.py
        reg_factor_scheduler = ExponentialScheduler(reg_factor_init, x_final=reg_factor_final, n_epoch=n_epoch) # test for step_factor=1.000691014168259 ?
        simplex = Simplex(corpus_examples=corpus_inputs,
                corpus_latent_reps=corpus_latents)    # test for corpus_size=100 & dim_latent=50
        simplex.fit(test_examples=test_inputs,
                    test_latent_reps=test_latents,
                    n_keep=decompostion_size,  # how many weights we want to keep in the end; keep all now to compare to own  model
                    n_epoch=n_epoch,
                    reg_factor=reg_factor_init,
                    reg_factor_scheduler=reg_factor_scheduler)      # test for ?
        weights = simplex.weights   # test for shape=10,100 & requires_grad=False
        # see mnist.py approximation_quality
        latent_rep_approx = simplex.latent_approx()     # test for shape=10,50 & requires_grad=False
        latent_rep_true = test_latents  # test for shape=10,50 & requires_grad=False
        
        # test unit -> if sum of top x weihgt adds up to at least ~90? or some other value a well trained original would have

        jacobian = simplex.jacobian_projection(test_id=test_id, model=classifier, input_baseline=input_baseline)


    if model_type is Model_Type.REIMPLEMENTED:
        print(f"Starting on cv {cv} with the reimplemented model!")
        scheduler = ExponentialScheduler(x_init=0.1, x_final=reg_factor_final, n_epoch=n_epoch)
        reg_factor = reg_factor_init

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
        # end of fit
            
        weights = torch.softmax(W_0, dim=-1).detach()
        weights = weights[test_id].numpy() # only single test item out of batch
        sort_id = np.argsort(weights)[::-1]

        corpus_decomposition = []
        
        for i in sort_id:
            corpus_decomposition.append((weights[i], corpus_inputs[i]))     # error: index 74 is out of bounds for axis 0 with size 10

        jacobian = []
    
    if model_type is Model_Type.TORCH_CONV:
        print(f"Starting on cv {cv} with our own model!")
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
        latent_rep_true = test_latents

        # jacobian does not work yet
        jacobian = []
    
    output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_r2_score = sklearn.metrics.r2_score(
        output_true, output_approx
    )
    latent_r2_score = sklearn.metrics.r2_score(
                latent_rep_true, latent_rep_approx.detach().numpy()
            )
    
    most_important_example = weights[0].argmax()  # for test example 0
    fig1 = plot_mnist(test_inputs[0][0], "Test example 0")
    fig2 = plot_mnist(corpus_inputs[most_important_example][0], f"M.i. attribution to example 0 (corpus id {most_important_example})")
    fig1.show()
    fig2.show()
    # BUG! There is something wildly wrong with my own model -> the most important example is (mostly) hugely different from the test example and shows some other number!
    # probably overfitting - or I need some way to get sparsity -> the according heighest weights are also super low, e.g. 0.01 vs 0.5 from original model
    # maybe thats also the regulator at play for the simplex model
    # try to play with weight decay --> want to keep weights sparse!
    # model is right now good at representation, but the weights are equally destributed
    print(f"Biggest contributor to test example 0: {weights[0].argmax()} with weight {weights[0].max()}")
    print(weights[0])
    # BUG! comparing the biggest contributor for test example 0 for original model and our model, it is never the same
    # But if definitely should be .... shouldn't it? they are not shuffeled in any way
    # if shuffle=False in corpus- and test-loader, the get the same results (but ir's logically also always the same for all cvs)!
    return weights, latent_r2_score, output_r2_score, jacobian


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

def do_mnist_experiment(cv):
    # the following are standard values which are used in mnist.py to train mnistClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #save_path="../experiments/results/mnist/quality/" #Jasmin
    save_path="IML/project-jml-project/files/" # Meike
    random_seed=42
    cv = 0 #1  # also for seeding
    model_reg_factor=0.1

    if not os.path.isfile(os.path.join(save_path,f"model_cv{cv}.pth")):
        
        train_model(
                device=device,
                random_seed=random_seed,
                cv=cv,
                save_path=save_path,
                model_reg_factor=model_reg_factor,
            )
    model = MnistClassifier()
    model.load_state_dict(torch.load(os.path.join(save_path,f"model_cv{cv}.pth")))
    model.to(device)
    model.eval()       

    # # Load corpus and test inputs - from toy example in Readme
    # corpus_loader = load_mnist(subset_size=100, train=True, batch_size=100) # MNIST train loader
    # test_loader = load_mnist(subset_size=10, train=True, batch_size=10) # MNIST test loader
    # corpus_inputs, _ = next(iter(corpus_loader)) # A tensor of corpus inputs; shape: ([100, 1, 28, 28])
    # test_inputs, _ = next(iter(test_loader)) # A set of inputs to explain; shape: ([10, 1, 28, 28])

    # # Compute the corpus and test latent representations - from toy example in Readme
    # corpus_latents = model.latent_representation(corpus_inputs).detach() # shape: ([100, 50]) - the 50 corresponds to  self.fc1 in init of MnistClassifier
    # test_latents = model.latent_representation(test_inputs).detach()  #shape: ([10, 50]) - the 50 corresponds to  self.fc1 in init of MnistClassifier

    # alternatively, get the loader from approximate_quality
    corpus_loader = load_mnist(100, train=True, shuffle=True)
    test_loader = load_mnist(10, train=False, shuffle=True)
    corpus_examples = enumerate(corpus_loader)
    test_examples = enumerate(test_loader)
    batch_id_test , (test_data, test_targets) = next(test_examples)
    batch_id_corpus, (corpus_data, corpus_target) = next(corpus_examples)
    corpus_data = corpus_data.to(device).detach()   #TODO: remove to device everywhere for consistency
    test_data = test_data.to(device).detach()
    corpus_latent_reps = model.latent_representation(corpus_data).detach()
    test_latent_reps = model.latent_representation(test_data).detach()

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
  

    # original_model, original_score_latent, original_score_output, original_jacobian = do_simplex(True, corpus_inputs, corpus_latents, test_inputs, test_latents, model, cv) #reg_factor_init, reg_factor_final, n_epoch, decompostion_size, test_id

    # own_model, own_score_latent, own_score_output, own_jacobian = do_simplex(False, corpus_inputs, corpus_latents, test_inputs, test_latents, model, cv) #reg_factor_init, reg_factor_final, n_epoch, decompostion_size, test_id
    
    # original_model, original_score_latent, original_score_output, original_jacobian
    original = do_simplex(Model_Type.ORIGINAL, corpus_data, corpus_latent_reps, test_data, test_latent_reps, model, cv, reg_factor_init, reg_factor_final, n_epoch, decompostion_size, test_id)

    # reimplemented_model, reimplemented_score_latent, reimplemented_score_output, reimplemented_jacobian
    reimplemented = do_simplex(Model_Type.REIMPLEMENTED, corpus_data, corpus_latent_reps, test_data, test_latent_reps, model, cv, reg_factor_init, reg_factor_final, n_epoch, decompostion_size, test_id)

    #own_model, own_score_latent, own_score_output, own_jacobian
    jasmin = do_simplex(Model_Type.TORCH_CONV, corpus_data, corpus_latent_reps, test_data, test_latent_reps, model, cv, reg_factor_init, reg_factor_final, n_epoch, decompostion_size, test_id)

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

    # with corpus loader etc. from toy example from readme 
    # Original score latents: [0.7946334516390705, 0.9502387168463264, 0.8776990118679626, -2.8175387750416063, 0.707885567216421, -1.815975287622799, 0.8881337347072017, 0.8852853046117132, 0.5199201748785078, 0.833508087622493]; mean: 0.1823789986725291
    # Original score outputs: [0.9470989250756885, 0.9824915604730456, 0.943139081448148, 0.9675954206884544, 0.9271998552154768, 0.9521581777271303, 0.9784162727330215, 0.9364662972726026, 0.9622678651950883, 0.9435156432248467]; mean: 0.9540349099053502
    # Own score latents: [0.9599862656593562, 0.9999853155903878, 0.9999917564391717, 0.999966531920891, 0.9798710758926213, 0.9986017927542699, 0.9999994139476136, 0.9999840061721373, 0.9599881614609406, 0.9998706580513061]; mean: 0.9898244977888695
    # Own score outputs: [0.9999980558177407, 0.9999989020126142, 0.9999999698333998, 0.9999999937027951, 0.9999957397955311, 0.9999998505979504, 0.9999999930597694, 0.9999994623653559, 0.9999997333445897, 0.9999971711228868]; mean: 0.9999988871652633
    # -> Our model is significantly better than the original one -> sanity check! something definitly seems up with the original one

    # with corpus loader etc. from approximate_quality function in  mnist.py
    # Original score latents: [0.9167337879979937, 0.9405107994734783, 0.91686397749137, 0.8771494016491066, 0.9046617856409526, 0.674657205391108, -0.5225561874228698, 0.9073619705563286, 0.9216722154508826, 0.9466629082920862]; mean: 0.7483717864520437
    # Original score outputs: [0.9755159978851694, 0.969792143498017, 0.9709427624041828, 0.9636141658403103, 0.9599413291786479, 0.9371682561049927, 0.9656454513180501, 0.9259867958673847, 0.9413008012745134, 0.9688911852716446]; mean: 0.9578798888642913
    # Own score latents: [0.9923508695956924, 0.9999856354516737, 0.9999883822535813, 0.9999957823629743, 0.9999892386561916, 0.9999636261664742, 0.9599113812028948, 0.9799918110751374, 0.9999659068803443, 0.9999773191522396]; mean: 0.9932119952797205
    # Own score outputs: [0.9999999527049385, 0.9999983486100354, 0.9999999082173288, 0.9999998679228843, 0.9999998338482943, 0.9999991961193568, 0.9999987473860438, 0.9999991518938114, 0.9999915755742068, 0.9999908166337439]; mean: 0.9999977398910644
    # -> much better, still strange;espeically vc=6 seems bad

    # something is up with the original model ... maybe play mit n_keep? I do the same as in approximate_quality, or don't I?
    # if shuffle=False in corpus- and test-loader, the original values get better (and more consistent)!

if __name__ == "__main__":
    # training mnist, from minst.py
    #run_multiple_experiments()
    do_mnist_experiment(6)
   
    

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