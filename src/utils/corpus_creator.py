from torch.utils.data import DataLoader
import torch
import random

def make_corpus(corpus_loader: DataLoader, corpus_size=100, n_classes=2):
    """
    incrementally samples from corpus loader, keeps class balance, performs reservoir sampling to sample with equal likelihood for every file
    """
    corpus_files = []
    labels = []
    labelk = [0]*n_classes

    for data, label in corpus_loader:# do reservoir sampling
        for file, label in zip(data, label):
            if(len(corpus_files)>=corpus_size):# if corpus is full replace random element with probability (corpus_size/2)/labelk[label]
                corpus_1_indices = [i for i,x in enumerate(labels) if x == label]
                labelk[label]+=1
                sample_index = random.choice(range(labelk[label]))
                if sample_index < len(corpus_1_indices):
                    corpus_files[corpus_1_indices[sample_index]]=file
                    labels[corpus_1_indices[sample_index]]=label
            else:     # fill corpus with files, keep class balance
                if labels.count(label)<corpus_size//n_classes:
                    corpus_files.append(file)
                    labels.append(label)
                    labelk[label]+=1
                else:
                    corpus_1_indices = [i for i,x in enumerate(labels) if x == label]
                    labelk[label]+=1
                    sample_index = random.choice(range(labelk[label]))
                    if sample_index < len(corpus_1_indices):
                        corpus_files[corpus_1_indices[sample_index]]=file
                        labels[corpus_1_indices[sample_index]]=label

    corpus_files = torch.stack(corpus_files)    
    labels = torch.stack(labels)

    return [corpus_files, labels]

