from captum.attr._utils.visualization import visualize_image_attr
import enum
import numpy as np
import torch

import evaluation as e
import simplex_versions as s
import classifier_versions as c


RANDOM_SEED=42

class Model_Type(enum.Enum):
    ORIGINAL = 1
    ORIGINAL_COMPACT = 2
    REIMPLEMENTED = 3
    R_NO_SOFTMAX = 4
    O_C_NO_SOFTMAX = 5
    O_C_NO_REGULIZATION = 6


class Dataset(enum.Enum):
    MNIST = 1
    OTHER = 2


# code mostly from the toy example from their github page right now,  also referencing use_case.py

# when using another dataset, we also have to replace load_minst (with our custom loader) and MnistClassifier (maybe with another pretrained model)

# for this to work right now you have to follow the install instructions of the original repo


def do_simplex(model_type=Model_Type.ORIGINAL, dataset=Dataset.MNIST, cv=0, decompostion_size=100, test_id=0, print_jacobians=False):
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
    torch.random.manual_seed(RANDOM_SEED + cv)
    torch.backends.cudnn.deterministic = True # https://www.typeerror.org/docs/pytorch/generated/torch.nn.conv1d

    # must return classifier, corpus_data, corpus_latents, corpus_target, test_data, test_targets, test_latents
    if dataset is Dataset.MNIST:
        classifier, corpus, test_set = c.train_or_load_mnist(RANDOM_SEED, cv)
        #TODO: maybe keep as triples and hand triples to models
        corpus_data, corpus_latents, corpus_target = corpus
        test_data, test_targets, test_latents = test_set

    # must return latent_rep_approx, weights, jacobian
    if model_type is Model_Type.ORIGINAL:
        print(f"Starting on cv {cv} with the original model!")
        latent_rep_approx, weights, jacobian = s.original_model(corpus_data, corpus_latents, test_data, test_latents, decompostion_size, test_id, classifier)

    if model_type is Model_Type.ORIGINAL_COMPACT:
        print(f"Starting on cv {cv} with the compact original model!")
        latent_rep_approx, weights, jacobian = s.compact_original_model(corpus_data, corpus_latents, test_data, test_latents, decompostion_size, test_id, classifier)
        
    if model_type is Model_Type.REIMPLEMENTED:
        print(f"Starting on cv {cv} with our own reimplemented model!")
        latent_rep_approx, weights, jacobian = s.reimplemented_model(corpus_data, corpus_latents, test_data, test_latents, decompostion_size, test_id, classifier)

    latent_rep_true = test_latents  # test for shape=10,50 & requires_grad=False
    
    output_r2_score, latent_r2_score = e.r_2_scores(classifier, latent_rep_approx, latent_rep_true, weights, test_id, test_data, corpus_data)
    
    #TODO: use latents instead of weights?
    decompostions = e.create_decompositions(test_data, test_targets, corpus_data, corpus_target, decompostion_size, weights)

    #TODO: how to print jacobians? (saliency is tensor shape[28,28,1])
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

#  auswertungen:
# 1. r2 repr (jasmin)
# 2. welche bilder sind die top x erklärungen, zahl, id, weights
# doch nicht: # 2. r2 score zahl (jasmin)


def do_mnist_experiment(cv=0):
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
    original = do_simplex(Model_Type.ORIGINAL, Dataset.MNIST, cv, decompostion_size, test_id)

    # reimplemented_model, reimplemented_score_latent, reimplemented_score_output, reimplemented_jacobian
    #meike = do_simplex(Model_Type.ORIGINAL_COMPACT, Dataset.MNIST, cv, decompostion_size, test_id)

    #own_model, own_score_latent, own_score_output, own_jacobian
    #jasmin = do_simplex(Model_Type.REIMPLEMENTED, Dataset.MNIST, cv, decompostion_size, test_id)

    #print(original[1], original[2]) # for cv=1 : 0.9178502448017772 0.9458505321920692
    #print(meike[1], meike[2]) # for cv=1 : 0.9178502448017772 0.9458505321920692
    #print(jasmin[1], jasmin[2]) # for cv=1 :0.9999924820980085 0.999999255618875

    return 

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

def run_all_experiments(corpus_size, test_size, decomposition_size, ):
    
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