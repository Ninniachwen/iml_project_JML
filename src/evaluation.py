import sklearn
import torch

def r_2_scores(classifier, latent_rep_approx:torch.Tensor, latent_rep_true:torch.Tensor) -> list[float]:
    """
    Calculates R2 scores between two latent representations. Same as in original Simplex.

    Args:
        classifier (MNIST | CatsAndDogs | HeartFailure): One of the three trained classifiers
        latent_rep_approx (torch.Tensor): the latent representation from this classifier
        latent_rep_true (torch.Tensor): the true latent representation 

    Returns:
        list[float]: R2 score
    """
    output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_r2_score = sklearn.metrics.r2_score(
        output_true, output_approx
    )
    latent_r2_score = sklearn.metrics.r2_score(
                latent_rep_true, latent_rep_approx.detach().numpy()
            )

    return output_r2_score, latent_r2_score


def create_decompositions(test_data:torch.Tensor, test_targets:torch.Tensor, corpus_data:torch.Tensor, corpus_targets:torch.Tensor, decompostion_size:int, weights:torch.Tensor, model_type:str, dataset:str) -> list[dict]:#TODO uplate description or remove model_type and dataset
    """
    collects decomposition for each test sample, using the top n corpus images. n = decomposition_size. eg: {'sample_id': 0, 'img': tensor(...), 'target': 7, 'decomposition': [ {'c_id': 84, 'c_weight': 0.79606116, 'c_img': tensor(...), 'c_target': 7}, ... ]}

    Args:
        test_data (torch.Tensor): Feature vector of Test examples (not used, but given as argument to keep function calls the same)
        test_targets (torch.Tensor): Targets (labels) of Test examples
        corpus_data (torch.Tensor): Feature vector of Corpus examples
        corpus_targets (torch.Tensor): Targets (labels) of Corpus examples
        decompostion_size (int): with how many corpus examples a test example should be explained
        weights (torch.Tensor): weights of the corpus examples
        model_type (str): The type of simplex model; see Enum Model_Type in main.py
        dataset (str): which dataset to use; see Enum Dataset  in main.py

    Returns:
        list[dict]: list of decompositions, which are structured in a nested dictionary each.
    """

    full_decomposition = []
    for s_id, sample, target in zip(range(len(test_data)), test_data, test_targets):
        sample_weights = weights[s_id].numpy()
        top_x_ids:list[int] = sample_weights.argsort()[-decompostion_size:][::-1]

        if(len(corpus_data.shape) == 4):
            size = [sample.shape[1], sample.shape[2]]
        else:
            size = [sample.shape[0]]

        decompostion = []
        for w_id in top_x_ids:
            decompostion.append({
                "c_id":w_id,
                "c_weight": sample_weights[w_id], 
                "c_img": corpus_data[w_id].reshape(size),
                "c_target": corpus_targets[w_id].item()})
        
        per_sample = {"sample_id": s_id,
                      "model_type": model_type,
                      "dataset": dataset,
                      "img" : sample.reshape(size),
                      "target" : target.item(),
                      "decomposition" : decompostion
                      }

        full_decomposition.append(per_sample)

    return full_decomposition