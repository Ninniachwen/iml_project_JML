[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/k0DpfI3g)
# IML WS 23 Project


## Introduction

For our reimplementation project, we chose the paper '*Explaining Latent Representations with a Corpus of Examples*' by Jonathan Crabbe, Zhaozhi Qian, Fergus Imrie, and Mihaela van der Schaar. 

The approach introduces **SimplEx**, which aims to give insights into a blackbox model by providing example-based explanations. This is done by returning a set of weighted examples from a given corpus of exemplary inputs. The weights can be interpreted as percentages that represent the similarities (and therefore the importance) of the respective corpus example to a chosen test example.

The code has been published on Github and is available at https://github.com/JonathanCrabbe/Simplex.

Their repository can be found in the folder `original_code`. While trying to keep it in its original condition, we had to change some paths to be able to reference them correctly in our code.

## Code Overview
```
project-jml-project
├── data                                # directory for the used datasets
│   ├── Animal Images                   # directory for Cats and Dogs dataset, source:TODO
│   ├── heart.csv                       # Heartfailure dataset, source:TODO
│   └── MNIST                           # directory for MNIST dataset, source: "http://yann.lecun.com/exdb/mnist/"
├── files
│   ├── ablation_results_original.csv       # results of ablation study
│   ├── approximation_quality_results.csv   # results of recreating the MNIST Approximation Quality Experiment(see mnist.py in original code)
│   ├── comparison_results.csv              # results of comparison between original and reimplemented model using diverse datasets and settings
│   ├── images                              # directory for created images
│   ├── models                              # directory for trained and saved blackbox classifier models
├── original_code                       # directory of the repository of the original code
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
│   ├── simplex_versions.py             # training of the different simplex version (original and reimplemented)
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

`pip install -r requirments.txt`

## Reimplementation

We reimplemented the model by introducing the class `Simplex_Model`, inheriting from torch.nn.Module. Our idea was to make the training of the Simplex model (which can be found in `original_code/src/simplexai/explainers/simplex.py`) more intuitive than the original, where the training was done without using predefined methods like "forward". 

The original simplex method has a parameter `n_keep` which is used for the decomposition size (how many corpus examples should be used for the explanation of the test example). Internally, it is used for regularization. In our reimplementation, we train the model independent of the decomposition size and later on set the weights of the corpus examples which should not be in the decomposition to zero. 

The reimplemented model can be found in file `src/simplex_versions.py` . The training is done in function `reimplemented_model`.

## Evaluation on Original Dataset

The paper did experiments on two datasets: MNIST (images) and Prostate Cancer (tabular data). We chose to evaluate our model and compare it using the MNIST dataset.

For metrics, we chose the same as the paper: the r2 scores of the latent representation and of the output. 

TODO weiter

## Evaluation on New Dataset

We evaluate on two new datasets, for which we each train a new black box classifier : The cats and dogs dataset:

`https://www.kaggle.com/datasets/unmoved/30k-cats-and-dogs-150x150-greyscale`

and the heart failure predicitons dataset:

`https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data`

The cats and dogs classifer is a CNN with three convolutional and two linear layers, therefore having one more layer than the original papers MNIST Classifier. 
The other model is a 4-layer linear neural network. 
Model training for one cats and dogs classifer over 40 epochs takes roughly 2 hours, therefore pre-trained models are provided in files/models. They each achieve a test accuracy of roughly 87%.

## Extensions of the Approach

We extended the apporach by providing an automatic corpus creator, that samples incrementally from a provided dataloader and provides a corpus with class balance.
Furhtermore we provide a visual decomposition for the mnist and cats and dogs classifiers from the experiments. They can be found in files/images.


## Ablation Study
### How to run
To execute the tests for the ablation study, run the following command in the root directory:

`python3 src/main.py -ablation`

The results will be written to the file `/src/files/ablation_results.csv`

The original results of our ablation study can be seen in file `/src/files/ablation_results_original.csv`.
### Experiment setup
We created the following models and modified versions of them to test with different parameters (see below):
1. original model 
2. original compact model (*we rewrote the original code to a more compact form for ablation purposes*)
3. reimplemented model
4. model #3 without softmax layer
5. model #3 with a normalization\* instead of the softmax layer
6. model #3 with randomized initial weights (*instead of initialized with zero*)
7. model #4 with randomized initial weights
8. model #5 with randomized initial weights 
9. model #2 without softmax layer
10. model #2 without regularization

\* in each epoch, we devide the weights of the corpus examples for each test example with the sum of the according weights so they add up to 1.

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
* the used paramenters
* the achieved latent and output r2 scores
* the id of the most important corpus example (according the the chosen simplex model) for the given test id and the correspondig weights
* the ids of the other important corpus examples (given by decomposition size), sorted by weights descendingly
* the targets of all the important corpus examples and the target of the test example
  
Especially the with last point, we can observe how good the given explanations are. The intuition: the most important corpus examples should be the same target as our test example we want to explain, otherwise the simplex model is probably not that great (or the examples really look alike).

### Results
First, we observe that the original model and the compact original model without any changes achieve the exactly same r2 scores and weights. This makes sense as the code is the same and the comparison is done as a sanity check.

We can observe that the models without a softmax layer (models #4, #7  and #9) seeimgly overfit during training. When using the whole corpus size for the explanation (corpus size = decomposition size), the r2 scores are over 0.99. If the decomposition size is smaller, the score drops below zero.  
When using our own reimplementations (model #4, #7), the weight of the heighest contributing corpus example however is extremely small, mostly little over 1/(corpus size). When using the compact original model, the weights are higher because of the used regularization. In all three cases, however, the most important corpus examples for explaining the test example have mostly the wrong label and the according ids are not identical to the ones from the original model. The model with the randomly initialized weights (#7) seems to be worse than the other two.  
The bad performance of omiting the softmax layer makes sense as the distribution of the weights to add up to 1 after each epoch is missing.

...

kurz zusammengefasst (später ausführlicher)
* no softmax original model compact hat insgesamt höhere einzelne weights als das von unserer reimpmlementation (vermutlich wegen regularisierung) aber die erklärenden bilder sind immer teilweise schlecht (TODO genauer nochmal verlgeichen- manchmal sind top x bilder auch okay und nach an original)
* normalize layer mit 0 weights trainiert überhaupt nicht - weight immer 1/corpus size, r2 immer schlecht
* normalize layer mit random weights scheinbar wie no softmax, overfitted; gewichte höher als ohne softmax (nochmal überprüfen)
* reimplemented model mit randomisierten inistialen weights ist etwa genauso gut wie mit zu 0 gesetzen weights. 
* original model ohne regularisierung: r2 immer gleich weil gewichte immer gleich sind.
* reimplemented model bei decomposition size = corpus size genau gleich wie original. Bei decompsition size < corpus size sind die r2 werte etwas schlechter (~ 3 Prozentpunkte?, genauer checken) und die gewichte vom original höher -> vermutlich wegen der regularisierung. TODO: das gehört (auch) zum Punkt "vergleich reimplemented vs original"
## Testing

### Unit tests
to execute unit tests, run `python -m tests.unittests` in root directory

## Technical 

The ablation study was done on a hp-Elitebook with a 11th Gen Intel® Core™ i7-1185G7 @ 3.00GHz × 8 CPU running Ubuntu 20. An experiment run with all 10 models types used for the ablation study took circa 30 seconds. The overall 560 combinations took about 40 minutes. There was no notable differente between training times of the different models.


TODO: future works: join test_set into corpus and see what happens
TODO: mention jacobian comparison score
TODO: mention self explaining corpus
