from torch.utils.data import DataLoader
from typing import Tuple
import torch
import random

def make_corpus(corpus_loader: DataLoader, corpus_size=100, n_classes=2, random_seed=None) -> Tuple[torch.Tensor,torch.Tensor]:
    """Creates a corpus from a dataloader. Samples are choosen uniformly random

    Args:
        corpus_loader (DataLoader): dataloader, with corpus data to sample.
        corpus_size (int, optional): size of the corpus. Defaults to 100.
        n_classes (int, optional): number of classes. If it exceed the amount of files in the dataloader it sets to the size of the dataset in the dataloader. Defaults to 2.
        random_seed (_type_, optional): random seed for reproducibility. Defaults to None.

    Returns:
        List: _description_
    """
    if random_seed:
        random.seed(random_seed)

    if corpus_size > len(corpus_loader.dataset): 
        corpus_size = len(corpus_loader.dataset)
    corpus_files = []
    labels = []
    labelk = [0]*n_classes

    for data, label in corpus_loader: # does reservoir sampling  #TODO creates UserWarning: resize transforms (use antialias=True)
        for file, label in zip(data, label): # iterates over all files and labels
            labelk[label]+=1
            if(len(corpus_files)>=corpus_size):# if corpus is full replace random element with probability (corpus_size/num_classes)/labelk[label]
                corpus_1_indices = [i for i,x in enumerate(labels) if x == label]
                sample_index = random.choice(range(labelk[label]))
                if sample_index < len(corpus_1_indices):
                    corpus_files[corpus_1_indices[sample_index]]=file
                    labels[corpus_1_indices[sample_index]]=label
                    
            else:     # fill corpus with files, keep class balance
                if labels.count(label)<((corpus_size+1)//n_classes):#
                    corpus_files.append(file)
                    labels.append(label)
                else:
                    corpus_1_indices = [i for i,x in enumerate(labels) if x == label]
                    sample_index = random.choice(range(labelk[label]))
                    if sample_index < len(corpus_1_indices):
                        corpus_files[corpus_1_indices[sample_index]]=file
                        labels[corpus_1_indices[sample_index]]=label
                        
    corpus_files = torch.stack(corpus_files)    
    labels = torch.stack(labels)
    labels = labels.reshape(shape=(corpus_size,))
    return (corpus_files, labels.to(torch.int64))

