import torch


def fdr(true_labels, noisy_labels, preds):  # preds has same shape as labels, it's a boolean tensor saying
                                            # True when it thinks it's looking at a non-corrupted label and
                                            # False when it thinks it's looking at a corrupted label
                                            # all tensors have size (nx1)
    true_preds = true_labels == noisy_labels
    true_positives = torch.sum((preds == True) & (true_preds == True)).item()
    false_positives = torch.sum((preds == True) & (true_preds == False)).item()
    fdr_value = false_positives / (true_positives + false_positives)
    return fdr_value


def recall(true_labels, noisy_labels, preds):
    true_preds = true_labels == noisy_labels
    true_positives = torch.sum((preds == True) & (true_preds == True)).item()
    false_negatives = torch.sum((preds == False) & (true_preds == True)).item()
    recall_value = true_positives / (true_positives + false_negatives)
    return recall_value

def precision(true_labels, noisy_labels, preds):
    true_preds = true_labels == noisy_labels
    true_positives = torch.sum((preds == True) & (true_preds == True)).item()
    false_positives = torch.sum((preds == True) & (true_preds == False)).item()
    precision_value = true_positives / (true_positives + false_positives)
    return precision_value

def f1(true_labels, noisy_labels, preds):
    prec = precision(true_labels, noisy_labels, preds)
    rec = recall(true_labels, noisy_labels, preds)
    f1_value = (2 * prec * rec) / (prec + rec)
    return f1_value

