from torch.utils.data import DataLoader
import torch
import random

def make_corpus(corpus_loader: DataLoader, corpus_size=100, n_classes=2):
    """
    incrementally samples from corpus loader, keeps class balance, performs reservoir sampling to sample with equal likelihood for every file fore ach class in the loader.
    """

    if corpus_size > len(corpus_loader.dataset): 
        corpus_size = len(corpus_loader.dataset)
    corpus_files = []
    labels = []
    labelk = [0]*n_classes

    for data, label in corpus_loader: # do reservoir sampling
        for file, label in zip(data, label):
            labelk[label]+=1
            if(len(corpus_files)>=corpus_size):# if corpus is full replace random element with probability (corpus_size/2)/labelk[label]
                corpus_1_indices = [i for i,x in enumerate(labels) if x == label]
                sample_index = random.choice(range(labelk[label]))
                if sample_index < len(corpus_1_indices):
                    corpus_files[corpus_1_indices[sample_index]]=file
                    labels[corpus_1_indices[sample_index]]=label
                    
            else:     # fill corpus with files, keep class balance
                if labels.count(label)<corpus_size+1//n_classes:
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

    return [corpus_files, labels.to(torch.int64)]

