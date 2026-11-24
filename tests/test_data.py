import numpy as np
import torch

from eo_edge_zoo.data import subset_idx

# 3 classes, sizes 10/20/30
Y = np.repeat(np.arange(3), [10, 20, 30])


def test_subset_is_stratified_per_class():
    idx = subset_idx(Y, frac=0.1, seed=0, num_classes=3)
    assert np.bincount(Y[idx], minlength=3).tolist() == [1, 2, 3]


def test_subset_keeps_at_least_one_per_class():
    idx = subset_idx(Y, frac=0.01, seed=0, num_classes=3)
    assert np.bincount(Y[idx], minlength=3).tolist() == [1, 1, 1]


def test_full_fraction_returns_all_indices():
    assert subset_idx(Y, frac=1.0, seed=2, num_classes=3).tolist() == list(range(60))


def test_subset_matches_notebook_subsets():
    # literals captured from original notebook impl -> paired comparisons stay valid
    assert subset_idx(Y, 0.1, 0, 3).tolist() == [1, 18, 19, 46, 51, 58]
    assert subset_idx(Y, 0.2, 1, 3).tolist() == [4, 7, 13, 16, 20, 27, 30, 37, 47, 52, 56, 59]


def test_subset_accepts_torch_labels():
    assert subset_idx(torch.from_numpy(Y), 0.1, 0, 3).tolist() == [1, 18, 19, 46, 51, 58]
