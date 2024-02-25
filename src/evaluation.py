import sklearn
import torch

def r_2_scores(classifier, latent_rep_approx:torch.Tensor, latent_rep_true:torch.Tensor) -> list[float]:
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
    

    Args:
        test_data (torch.Tensor): _description_
        test_targets (torch.Tensor): _description_
        corpus_data (torch.Tensor): _description_
        corpus_targets (torch.Tensor): _description_
        decompostion_size (int): _description_
        weights (torch.Tensor): _description_

    Returns:
        list[dict]: _description_
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