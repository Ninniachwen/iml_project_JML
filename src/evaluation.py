import sklearn
import torch

#TODO: what type is classifier? one of three or only mnist?
def r_2_scores(classifier, latent_rep_approx:torch.Tensor, latent_rep_true:torch.Tensor) -> list[float]:
    #TODO: latent_to_presoftmax for other classifiers as well? (lucas)
    output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_r2_score = sklearn.metrics.r2_score(
        output_true, output_approx
    )
    latent_r2_score = sklearn.metrics.r2_score(
                latent_rep_true, latent_rep_approx.detach().numpy()
            )

    return output_r2_score, latent_r2_score


def create_decompositions(test_data:torch.Tensor, test_targets:torch.Tensor, corpus_data:torch.Tensor, corpus_targets:torch.Tensor, decompostion_size:int, weights:torch.Tensor) -> list[dict]:
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