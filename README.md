[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/k0DpfI3g)
# IML WS 23 Project


## Introduction

For our reimplementation project, we chose the paper '*Explaining Latent Representations with a Corpus of Examples*' by Jonathan Crabbe, Zhaozhi Qian, Fergus Imrie, and Mihaela van der Schaar. 

The approach introduces **SimplEx**, which aims to give insights into a blackbox model by providing example-based explanations. This is done by returning a set of weighted examples from a given corpus of exemplary inputs. The weights can be interpreted as percentages that represent the similarities (and therefore the importance) of the respective corpus example to a chosen test example. If the original sample is included in the corpus used to explain it, then this corpus sample receives, as expected, 99% importance in the decomposition.

The code has been published on Github and is available at https://github.com/JonathanCrabbe/Simplex.

Their repository can be found in the folder `original_code`. While trying to keep it in its original condition, we had to change some paths to be able to reference them correctly in our code.

## Code Overview
```
project-jml-project
├── data                                # directory for the used datasets
│   ├── Animal Images                   # directory for Cats and Dogs dataset, source https://www.kaggle.com/datasets/unmoved/30k-cats-and-dogs-150x150-greyscale
│   ├── heart.csv                       # Heartfailure dataset, source https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data
│   └── MNIST                           # directory for MNIST dataset, source: "http://yann.lecun.com/exdb/mnist/"
├── files
│   ├── ablation_results_original.csv                # results of ablation study
│   ├── approximation_quality_results_original.csv   # results of recreating the MNIST Approximation Quality Experiment(see mnist.py in original code)
│   ├── cats_and_dogs_results_original.csv           # results of testing simplex models on cats and dogs dataset
│   ├── heartfailure_original.csv                    # results of testing simplex models on heartfailure dataset
│   ├── mnist_results_original.csv                   # results of testing simplex models on mnist dataset
│   ├── comparison_results.csv                       # results of comparison between original and reimplemented model using diverse datasets and settings
│   ├── images                              # directory for created images
│   ├── classifier                          # directory for trained and saved blackbox classifier models
│   ├── original_experiment                 # directory for created plots for the MNIST Approximation Quality Experiment
│   ├── presentation                        # directory for the pictures used for our presentation
├── original_code                       # directory of the repository of the original SimplEx code
├── JML Projekt Poster final.pdf        # our presentation poster
├── README.md                           # this README
├── requirements.txt                    # the requirements for running the code
├── src                                 # directory for our code
│   ├── cats_and_dogs_training.py       # training of the blackbox classifier for the Cats and Dogs dataset
│   ├── classifier                      # directory for the classes of the classifier models
│   │   ├── CatsAndDogsClassifier.py
│   │   ├── HeartfailureClassifier.py
│   ├── classifier_versions.py          # training or loading the different classifier 
│   ├── datasets                        # directory for loading the different datasets
│   │   ├── cats_and_dogs_dataset.py
│   │   ├── heartfailure_dataset.py
│   ├── evaluation.py                   # methods to evaluate the different simplex models (e.g. r2scores)
│   ├── heartfailure_training.py        # training of the blackbox classifier for the heartfailure dataset
│   ├── main.py                         # main function
│   ├── simplex_versions.py             # training of the different simplex versions (original, compact and reimplemented)
│   ├── utils                           # directory of different used helper functions
│   └── visualization                   # directory for plotting functions
└── tests                               # directory for unittests
```
TODO Kontrolle ob alles so passt

## Set up the environment
Create a conda environment:

`conda create -n jml_simplex python=3.10`

Activate the environment:

`conda activate jml_simplex`

Install all required packages:

`pip install -r requirements.txt`

## Reimplementation

We reimplemented the model by introducing the class `Simplex_Model`, inheriting from torch.nn.Module. Our idea was to make the training of the Simplex model (which can be found in `original_code/src/simplexai/explainers/simplex.py`) more intuitive than the original, where the training was done without using predefined methods like "forward". 

The original simplex method has a parameter `n_keep` which is used for the decomposition size (how many corpus examples should be used to explain the test example). Internally, it is used for regularization. In our reimplementation, we train the model independently of the decomposition size and later set the corpus examples' weights, which should not be in the decomposition, to zero. 

The reimplemented model can be found in the file `src/simplex_versions.py`. The training is done in the function `reimplemented_model`.

To execute some tests on all datasets, execute the following command in the root directory:

`python3 src/main.py -all`.

The results will be saved in `files/comparison_results.csv`.

### Compact Simplex

We also condensed the original SimplEx model from the authors' github repo in a single function call, to make the ablation study easier. In our code, we called this model `compact original`. The results and learned weights of this model are identical to the original, which can also be seen in the ablation study. 

## Evaluation using Original Dataset

The paper experimented on two datasets: MNIST (images) and Prostate Cancer (tabular data). We chose to evaluate and compare our model using the MNIST dataset.

For metrics, we chose the same as the paper: the r2 scores of the latent representation (comparing the original latent representations of the blackbox classifier to the latent representation learned by the simplex model) and the output (comparing the output of the blackbox classifier using the original latent representations to the output of the blackbox model using the latent representation learned by the simplex model).

Results of the original model, the compact original and the reimplemented model with different parameters can be found in  `/files/mnist_results_original.csv`. To run the test, run the following command in the root directory:

`python3 src/main.py -mnist`

The results will be written to the file `/files/mnist_results.csv`.


The reimplemented model achieved similar results to the original model, which are good overall. We see differences when using small decomposition sizes as the regularization of the original model gets better results.



## Evaluation using New Dataset

We evaluate the simplex versions on two new datasets, for which we each train a new black box classifier: The cats and dogs dataset:

`https://www.kaggle.com/datasets/unmoved/30k-cats-and-dogs-150x150-greyscale`

and the heart failure predictions dataset:

`https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data`

The cats and dogs classifier is a CNN with three convolutional and two linear layers, therefore having one more layer than the original papers MNIST Classifier. 
The other model is a 4-layer linear neural network. 
Model training for one cats and dogs classifer over 40 epochs takes roughly 2 hours, pre-trained models are provided in files/models. They each achieve a test accuracy of roughly 87%.
It is an interesting data set as the input images are 150x150 pixels and are harder to classify as the pictures vary more in the positioning of the object to classify and more noise and background is present.

Results of the original model, the compact original and the reimplemented model with different parameters on the Cats and Dogs dataset can be found in  `/files/cats_and_dogs_results_original.csv`. To run the test, run the following command in the root directory:

`python3 src/main.py -cats_and_dogs`

The results will be written to the file `/files/cats_and_dogs_results.csv`.

Results for the Heartfailure Dataset can be found in `/files/heartfailure_original.csv`. To run the test, run the following command in the root directory:

`python3 src/main.py -heartfailure`

The results will be written to the file `/files/heartfailure.csv`.

## Extensions of the Approach
We extended the apporach by providing an automatic corpus creator, that samples incrementally from a provided dataloader and provides a corpus with class balance. It performs reservoir sampling to sample uniformly random.  

Furthermore we provide a visual decomposition for the mnist and cats and dogs classifiers from the experiments, that can automatically be generated. They can be found in files/images and some are visible on the poster.
They calculated weights of the corpus examples as well as the true prediction of the classifer, indicating the confidence of the classifer for the prediction are provided.   

Additionally, we created a comparison score to compare Jacobian matrices with one another. For each jacobian, we located the most important pixel location per row. Then we calculated the avarage difference between this max-index-list for the two jacobians (because if the max pixel is not identical, but at least still close together, this is better than if they were far apart). This avarage index difference describes a rough similarity between two jacobian representations. We used it in the Tests, to check if original and compact original simplex had identical jacobians, and if our reimplemented jacobians were at least similar. Which they were.


## Ablation Study
### How to run
To execute the tests for the ablation study, run the following command in the root directory:

`python3 src/main.py -ablation`

The results will be written to the file `/files/ablation_results.csv`

The original results of our ablation study can be seen in file `/files/ablation_results_original.csv`.
### Experimental Setup
We created the following models and modified versions of them to test with different parameters (see below):
1. Original model 
2. Original compact model (*we rewrote the original code to a more compact form for ablation purposes*)
3. Reimplemented model
4. Model #3 without softmax layer
5. Model #3 with a normalization\* instead of the softmax layer
6. Model #3 with randomized initial weights (*instead of initialized with zero*)
7. Model #4 with randomized initial weights
8. Model #5 with randomized initial weights 
9. Model #2 without softmax layer
10. Model #2 without regularization

\* in each epoch, we divide the weights of the corpus examples for each test example with the sum of the according weights so they add up to 1.

For the models without softmax or normalization layer, we apply a normalization after completion of the training so the weights add up to 1 and can be interpreted as percentages.

We tested these models with a combination of the following parameters, which leads to 560 combinations:
* corpus_size = [50, 100]
* test_size = [10, 50]
* decomposition_size = [5, 10, 50, 100] (*we made sure that corpus_size < decomposition_size*)
* cv = [0,1]
* test_id = [0,1] (*in resulting file renamed to sample_id*)


We tested the models on the MNIST dataset.

While running the tests, we document:
* the used model
* the used parameters
* the achieved latent and output r2 scores
* the id of the most important corpus example (according to the chosen simplex model) for the given test id and the corresponding weights
* the ids of the other important corpus examples (given by decomposition size), sorted by weights descendingly
* the targets of all the important corpus examples and the target of the test example
  
Especially the with last point, we can observe how good the given explanations are. The intuition: the most important corpus examples should be the same target as our test example we want to explain, otherwise the simplex model is probably not that great (or the examples really look alike).

### Results
First, we observe that the original model and the compact original model without any changes achieve the exactly same r2 scores and weights. This makes sense as the code is the same and the comparison is done as a sanity check.

We can observe that the models without a softmax layer (models #4, #7, and #9) seemingly overfit during training. When using the whole corpus size for the explanation (corpus size = decomposition size), the r2 scores are over 0.99. If the decomposition size is smaller, the score drops below zero.  
When using our own reimplementations (model #4, #7), the weight of the highest contributing corpus example however is extremely small, mostly a little over 1/(corpus size). When using the compact original model, the weights are higher because of the used regularization. In all three cases, however, the most important corpus examples for explaining the test example have mostly the wrong label and the according ids are not identical to the ones from the original model. The model with the randomly initialized weights (#7) seems to be worse than the other two.  
The bad performance of omitting the softmax layer makes sense as the distribution of the weights to add up to 1 after each epoch is missing.

When using a normalization layer instead of a softmax layer, we see that the model does not train when we initialize the weights with 0 (model #5). When initializing the models at random (model # 8), the model achieves good results with relatively high weights.

We observe that our reimplemented model with initialized random weights (model #6) seems to be as good as the one initilaized with zeros (model # 3).

Observing the results of the original model without regularization (model # 10), we see that the r2 values stays the same. This makes sense as th model is now trained independent of the decomposition size without compensating for this in any other way.

## Testing

To execute unit tests, run `python -m tests.unittests` in the root directory.

## Technical 

The ablation study was done on a hp-Elitebook with an 11th Gen Intel® Core™ i7-1185G7 @ 3.00GHz × 8 CPU, 16GB RAM running Ubuntu 20. An experiment run with all 10 model types used for the ablation study took circa 30 seconds. The overall 560 combinations took about 40 minutes. There was no notable difference between training times of the different models.
The original experiment (`python3 src/main.py -original`) with all in all 120 combination was also done on this computer. It took around 1 hour. 

The unittests were done on a Surface Pro 2018 with an 11th Gen Intel® Core™ i7-1165G7 @ 2.80GHz x 2, 32GB Ram running Windows 10. A complete unit test took 20 minutes, with all the classifiers beeing loaded from disk.

Cats and dogs classsifer training was performed with 11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz with 16GB Ram running Windows Home 11. As mentioned before trainign over 40 epochs took 2 hours for each model.

