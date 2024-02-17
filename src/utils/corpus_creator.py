from torch.utils.data import DataLoader
import torch

def make_corpus(corpus_loader: DataLoader, corpus_size=1000):
    """
    incrementally samples from corpus loader, keeps class balance
    """
    corpus_data = [torch.tensor(),torch.tensor()]

    for data, label in corpus_loader:# do reservoir sampling
        for file, label in zip(data, label):
            if(len(corpus_data)>=corpus_size):
                pass
            else:
                corpus_data[0].append(file)
                corpus_data[1].append(label)
        

    return corpus_data

