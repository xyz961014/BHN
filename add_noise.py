import torch
import math
import random


def symmetric(eta, labels, K):  # eta is proportion of noisy labels, K is number of classes
    n = labels.size(0)          # number of datapoints in the batch
    shuffled_indices = torch.randperm(n)
    shuffled_labels = labels[shuffled_indices]
    max_noisy_index = math.floor(eta*n)     # number of labels to corrupt
    for i in range(max_noisy_index):
        possible_labels = list(range(K))    # all possible labels, representing all K classes
        possible_labels.remove(int(shuffled_labels[i][0]))   # remove current label from list of possible new labels
        shuffled_labels[i][0] = random.choice(possible_labels)  # replace label with a new random one
    noisy_labels = shuffled_labels[torch.argsort(shuffled_indices)]      # unshuffling the labels
    return noisy_labels

def asymmetric(eta, labels, K):  # eta is proportion of noisy labels, K is number of classes
    n = labels.size(0)  # number of datapoints in the batch
    shuffled_indices = torch.randperm(n)
    shuffled_labels = labels[shuffled_indices]
    max_noisy_index = math.floor(eta * n)  # number of labels to corrupt
    labels_to_corrupt = shuffled_labels[0:max_noisy_index, :]
    shuffled_labels[0:max_noisy_index,:] = (labels_to_corrupt + 1) % K  # shift labels by one forward
    noisy_labels = shuffled_labels[torch.argsort(shuffled_indices)]  # unshuffling the labels
    return noisy_labels

def instanceDependent(eta, features, labels, K):
    n = features.size(0)    # number of datapoints in the batch
    d = features.size(1)    # number of dimensions
    q_vector = torch.randn(n,1) * 0.1 + eta  # probabilities for each label to be corrupted, not yet truncated
    q_vector = torch.clamp(q_vector, 0, 1)  # truncate
    w = torch.randn(d, K)   # weights for projection
    for i in range(n):
        p = features[i,:] @ w   # we will eventually sample the new label according to distribution in vector p
                                # mathematically, p has size (1xK), but is represented here with size 10,
        p[labels[i]] = float('-inf')
        p = q_vector[i] * torch.softmax(p, 0)
        p[labels[i]] = 1 - q_vector[i]
        new_label = torch.multinomial(p, num_samples=1).item()  # sampling according to distribution given by p
        labels[i] = new_label
    return labels


