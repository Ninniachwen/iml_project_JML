[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/k0DpfI3g)
# IML WS 23 Project

## Set up the environment
Create a conda environment:

`conda create env jml_simplex`

Activate the environment:

`conda activate jml_simplex`

Install all required packages:

`pip install -r requirments.txt`

Make sure that python3 is >= version 3.10 .

## Reimplementation

We reimplemented the model by introducing the class Simplex_Model, inheriting from torch.nn.Module. Our idea was to make the training of the Simplex model more intuitive than the original, where the training was done without using torch methods like "forward". 

The model can be found in file `src/simplex_versions.py` .

## Ablation Study
### How to run
To execute the tests for the ablation study, run the following command in the root directory:

`python3 src/main.py -ablation`

### Experiment setup
We created the following models and modified versions of them to test with different parameters (see below):
1. original model 
2. original compact model (*we rewrote the original code to a more compact form for ablation purposes*)
3. reimplemented model
4. model #3 without softmax layer
5. model #3 with a standard normalization instead of the softmax layer
6. model #3 with randomized initial weights (*instead of initialized with zero*)
7. model #4 with randomized initial weights
8. model #5 with randomized initial weights 
9. model #2 without softmax layer
10. model #2 without regularization

We tested these models with the follwing parameters:
TODO welche parameter wir schlussendlich nehmen

We tested the models on the MNIST dataset. TODO: auch auf den anderen?

When running the tests, we document:
* the used model
* the used paramenters
* the achieved latent and output r2 scores
* the id of the most important corpus example (according the the chosen simplex model) for the given test id and the correspondig weights
* the ids of the other important corpus examples (given by decomposition size), sorted by weights descendingly
* the targets of all the important corpus examples and the target of the test example
  
Especially the with last point, we can observe how good the given explanations are. The intuition: the most important corpus examples should be the same target as our test example we want to explain, otherweise the simplex model is probably not that great (or the examples really look alike).

With our reimplementation, we incorporate the decomposition size (ds) in the following: for each test example we leave the highest ds weights as they are and set the other weights to zero. This way, while computing the r2 score, we only use the the most important ds corpus examples, and the r2 score changes for the different decomposition sizes.  TODO das viellecht wo anders erwähnen?

### Results
We can observe that the models without a softmax layer with weights initialized with zeros (models #4 and #9) seeimgly overfit during training. When using the whole corpus size for the explanation (corpus size = decomposition size), the r2 score is over 0.99. If the decomposition size is smaller, the score drops below zer0.
When using our own reimplementation (model #3), the weight of the heighest contributing corpus example however is extremely small, mostly little over 1/(corpus size). This 
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


## Original Code

The authors Jonathan Crabbé et al. published their original code on GitHub: https://github.com/JonathanCrabbe/Simplex. Their repository can be found in the folder "original_code". We changed only some relative import paths so our models can work.

## Technical 

The ablation study was done on a hp-Elitebook with a 11th Gen Intel® Core™ i7-1185G7 @ 3.00GHz × 8 CPU running Ubuntu 20. An experiment run with all 10 models types used for the ablation study took circa 30 seconds. The 54 combinations took accordingly about 30 minutes. There was no notable differente between training times of the different models. TODO eventuell anpassen