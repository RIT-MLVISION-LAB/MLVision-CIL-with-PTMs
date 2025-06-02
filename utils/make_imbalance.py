import numpy as np
import torch
from torch.utils.data import Subset


def _extract_labels(dataset):
    """
    Internal helper: try to pull out all labels from `dataset` in one shot.
    Supports:
      1) dataset.targets  (e.g. many torchvision datasets)
      2) dataset.labels   (some custom datasets)
      3) falling back to iterating once over __getitem__ for small datasets.
    Returns:
      np.ndarray of shape (N,) containing integer class‐labels.
    """
    if hasattr(dataset, "targets"):
        lbls = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        lbls = np.array(dataset.labels)
    else:
        # Fallback: iterate once (potentially slow if dataset is large,
        # but typically OK for a “one‐time imbalance creation” step).
        lbls = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            lbls.append(int(y))
        lbls = np.array(lbls)
    return lbls


def make_inter_class_imbalance(dataset, beta):
    """
    Return a Subset of `dataset` where class‐frequencies follow a β‐power law:
      let C = number of distinct classes, and sort them by label.
      For class indexed i (0‐based in sorted order), target_count_i ∝ (i+1)^(-beta).
      Then we simply cap at the original available count for that class and subsample.
    Args:
      - dataset: any PyTorch Dataset where `dataset[i]` returns (x, y) and
                 where y ∈ {0,1,…,C-1}.  We attempt to read dataset.targets or dataset.labels,
                 or else fall back to iterating once over __getitem__ to get y.
      - beta (float ≥ 0): imbalance parameter. Larger β → more skew (i.e. “longer tail”).
                          If β=0, then every class keeps its full original count (no imbalance).
    Returns:
      - torch.utils.data.Subset containing exactly
         ∑_i ⌊N_max * (i+1)^(-β)⌋  indices, clipped so we never ask for more than
         the original number of samples of each class.
       Here N_max = (original maximum per‐class count).
    Example usage:
      ds_imbal = make_inter_class_imbalance(original_dataset, beta=1.0)
      loader = DataLoader(ds_imbal, batch_size=...)
    """
    labels = _extract_labels(dataset)
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
        # (rank+1) so that the “first” class (rank=0) gets N_max * 1^(-β) = N_max,
        # the second (rank=1) gets N_max * 2^(-β), etc.
        raw = N_max * ( (rank + 1) ** (-beta) )
        tgt = int(np.floor(raw + 1e-8))
        # But we cannot exceed what was originally there:
        target_counts[c] = min(tgt, orig_counts[c])

    # Now choose indices for each class:
    selected_indices = []
    for c in classes_sorted:
        cls_idx = np.where(labels == c)[0]
        np.random.shuffle(cls_idx)  # randomize before subsampling
        keep = cls_idx[: target_counts[c]]
        selected_indices.extend(keep.tolist())

    # Return a Subset of the original dataset
    return Subset(dataset, selected_indices)


def make_intra_task_imbalance(dataset, task_id, num_tasks, mode="ascending"):

    n = len(dataset)
  
    if mode == "ascending":
        frac = 0.1 + (task_id / (num_tasks - 1)) * (1.0 - 0.1)
    elif mode == "descending":
        frac = 1.0 - (task_id / (num_tasks - 1)) * (1.0 - 0.1)


    frac = float(frac)
    frac = max(0.1, min(1.0, frac))

    # Determine how many samples to keep
    target_count = int(np.floor(n * frac))
    target_count = max(target_count, 1)  # at least one sample

    # Randomly choose that many indices
    all_indices = np.arange(n)
    np.random.shuffle(all_indices)
    selected = all_indices[:target_count].tolist()

    return Subset(dataset, selected)


def count_classes(dataset):
    from collections import Counter
    counter = Counter()
    for idx in range(len(dataset)):
        _, _, label = dataset[idx]
        counter[int(label)] += 1
    return dict(counter)


def _extract_labels(dataset):
    """
    Given either a Dataset or a Subset, return a numpy array of length N (N = number of 
    examples *in that dataset*), containing the integer labels for each sample in order.
    
    Supports:
      - If dataset has .targets or .labels, we grab that full array/tensor and then,
        if it's a Subset, we sub‐index it by subset.indices.
      - Otherwise (no .targets/.labels), we fall back to iterating over __getitem__ (slow).
    
    Returns:
      np.ndarray of shape (N,).
    """
    # 1) If it's a Subset, recursively extract from the base and then sub‐select
    if isinstance(dataset, Subset):
        base = dataset.dataset                   # the underlying dataset
        indices = np.array(dataset.indices)      # list of ints
        # Extract all labels from base (recursively)
        all_labels = _extract_labels(base)       # this returns a numpy array of shape (len(base),)
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