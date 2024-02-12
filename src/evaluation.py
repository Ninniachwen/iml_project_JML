
#TODO: create test for decomposition

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# directly from original code visualization/images.py
def plot_mnist(data, title: str = "") -> plt.Figure:
    fig = plt.figure()
    plt.imshow(data, cmap="gray", interpolation="none")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return plt

def r_2_scores(classifier, latent_rep_approx, latent_rep_true, weights, test_id, test_data, corpus_data, debugging=False):
    output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_r2_score = sklearn.metrics.r2_score(
        output_true, output_approx
    )
    latent_r2_score = sklearn.metrics.r2_score(
                latent_rep_true, latent_rep_approx.detach().numpy()
            )
    
    if debugging:
        #TODO: probably remove to elsewhere
        most_important_example = weights[test_id].argmax()  # for test example 0
        fig1 = plot_mnist(test_data[test_id][0], f"Test example {test_id}")
        fig2 = plot_mnist(corpus_data[most_important_example][0], f"M.i. attribution to example 0 (corpus id {most_important_example})")
        fig1.show()
        fig2.show()

        print(f"Biggest contributor to test example 0: {weights[0].argmax()} with weight {weights[0].max()}")
        print(weights[0])

    return output_r2_score, latent_r2_score


def create_decompositions(test_data, test_targets, corpus_data, corpus_targets, decompostion_size, weights):
    decompostion_size = 5

    full_decomposition = []
    for s_id, sample, target in zip(range(len(test_data)), test_data, test_targets):
        #assert test_id < test_size
        sample_weights = weights[s_id].numpy()
        top_x_ids = sample_weights.argsort()[-decompostion_size:][::-1]

        decompostion = []
        for w_id in top_x_ids:
            decompostion.append({
                "c_id":w_id,
                "c_weight": sample_weights[w_id], 
                "c_img": corpus_data[w_id].reshape([sample.shape[1], sample.shape[2]]), 
                "c_target": corpus_targets[w_id].item()})
        
        per_sample = {"sample_id": s_id,
                      "img" : sample.reshape([sample.shape[1],sample.shape[2]]),
                      "target" : target.item(),
                      "decomposition" : decompostion
                      }

        full_decomposition.append(per_sample)

    return full_decomposition