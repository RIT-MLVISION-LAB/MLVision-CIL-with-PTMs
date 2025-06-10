import numpy as np
import torch
from torch.utils.data import Subset


def extract_labels(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset                   # the underlying dataset
        indices = np.array(dataset.indices)      # list of ints
        # Extract all labels from base (recursively)
        all_labels = extract_labels(base)       # this returns a numpy array of shape (len(base),)
        # Now pick only those at `indices` (and preserve their ordering as stored in subset.indices)
        return all_labels[indices]

    # 2) If it has .targets or .labels, grab that:
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        # 3) Fallback: run through __getitem__ once (slow if very large)
        labels_list = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            labels_list.append(int(y))
        return np.array(labels_list, dtype=int)

    # At this point `labels` lives in the top‐level dataset, not a Subset
    # Convert to numpy:
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    else:
        labels = np.array(labels)
    return labels


def make_inter_class_imbalance(dataset, beta):

    labels = extract_labels(dataset)
    classes = np.unique(labels)
    classes_sorted = np.sort(classes)
    num_classes = len(classes_sorted)

    # Count how many examples we originally have per class:
    orig_counts = {c: int((labels == c).sum()) for c in classes_sorted}
    N_max = max(orig_counts.values())

    # Compute “target” number for each class under the power‐law:
    #   target_count[c] = floor( N_max * (rank(c)+1)^(-beta) ), clipped to orig_counts[c].
    target_counts = {}
    for rank, c in enumerate(classes_sorted):
        raw = N_max * ( (rank + 1) ** (-beta) )
        tgt = int(np.floor(raw + 1e-8))
        target_counts[c] = min(tgt, orig_counts[c])

    selected_indices = []
    for c in classes_sorted:
        cls_idx = np.where(labels == c)[0]
        np.random.shuffle(cls_idx)  # randomize before subsampling
        keep = cls_idx[: target_counts[c]]
        selected_indices.extend(keep.tolist())
    return Subset(dataset, selected_indices)


def make_intra_task_imbalance(dataset, task_id, num_tasks, mode="ascending"):

    n = len(dataset)
    task_id = (int(task_id)+19)%20
    if mode == "ascending":
        frac = 0.1 + (task_id / (num_tasks - 1)) * (1.0 - 0.1)
    elif mode == "descending":
        frac = 1.0 - (task_id / (num_tasks - 1)) * (1.0 - 0.1)

    target_count = max(int(n * float(frac)), 1)
    all_indices = np.arange(n)

    np.random.shuffle(all_indices)
    selected = all_indices[:target_count].tolist()
    print(f"Selected {len(selected)} samples from {len(all_indices)}) which is {len(selected)/len(all_indices)}")
    return Subset(dataset, selected)


def reduce_training_dataset(dataset):
    n = len(dataset)
    target_count = max(int(n * float(11/20)), 1)
    all_indices = np.arange(n)

    np.random.shuffle(all_indices)
    selected = all_indices[:target_count].tolist()
    print(f"Selected {len(selected)} samples from {len(all_indices)}) which is {len(selected)/len(all_indices)}")
    return Subset(dataset, selected)



def count_classes(dataset):
    from collections import Counter
    counter = Counter()
    for idx in range(len(dataset)):
        _, _, label = dataset[idx]
        counter[int(label)] += 1
    return dict(counter)