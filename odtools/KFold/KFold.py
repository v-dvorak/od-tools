import shutil
from itertools import combinations
from pathlib import Path
from random import Random

import numpy as np
from tqdm import tqdm

from ..Conversions.Formatter import _load_data_from_paths, _setup_split_dirs


def _to_k_folds(n: np.int64, k: np.int64, seed: np.int64 = 42) -> np.ndarray:
    """
    Generates list of K folds based on ``n`` and ``k``,
    where ``n`` is the length of data and ``k`` is the number of folds.
    """
    indexes = np.arange(n)
    Random(seed).shuffle(indexes)

    folds = [[] for _ in range(k)]
    for index in indexes:
        folds[index % k].append(index)

    return np.array(folds, dtype=np.ndarray)


def _leave_p_out(
        n: np.int64,
        k: np.int64 = 5,
        p: np.int64 = 1,
        verbose: bool = False,
        seed=42
) -> np.ndarray:
    """
    Implements Leave-P-Out (LPO) Cross-Validation.
    Generates K folds based on ``n``, ``k`` and ``p``,
    where ``n`` is the length of data, ``k`` is the number of folds
    and ``p`` is the number of folds that are left out.

    :param n: length of data
    :param k: number of folds
    :param p: number of folds that are left out
    :param verbose: make script verbose
    :param seed: seed for data shuffling
    :return: array of data indexes as (train, test)
    """
    k_folds = _to_k_folds(n, k, seed=seed)
    results = []

    for test_fold_indexes in combinations(range(len(k_folds)), p):
        test_subset = np.concatenate(
            [k_folds[index] for index in test_fold_indexes],
            dtype=np.int64
        )
        train_subset = np.concatenate(
            [k_folds[index] for index in range(len(k_folds)) if index not in test_fold_indexes],
            dtype=np.int64
        )
        results.append([train_subset, test_subset])

    if verbose:
        for i, (train, test) in enumerate(results):
            print(f"Split {i + 1}:\n  Train: {train}\n  Test: {test}\n")

    return np.array(results, dtype=np.ndarray)


def kfold_dataset(
        output_dir: Path,
        image_source_dir: Path,
        annot_source_dir: Path,
        folds: int = 5,
        pout: int = 1,
        seed: int = 42,
        verbose: bool = False,
) -> None:
    """
    Splits images and annotations into folds and assembles them to train and test sets
    based on the number of folds chosen and folds left out (= test set).

    For each dataset a folder with its index will be created
    with images and annotations split to train and val folders.

    Given data are split based regardless of their format.
    Config file will not be created.

    :param output_dir: path to save dataset to
    :param image_source_dir: path to image directory
    :param annot_source_dir: path to annotation directory
    :param folds: number of folds
    :param pout: Leave-P-Out, how many folds will be held out as test set
    :param seed: random seed
    :param verbose: make script verbose
    """
    data = _load_data_from_paths(
        image_source_dir, "*",
        annot_source_dir, "*"
    )
    data = np.array(data, dtype=tuple)

    folds = _leave_p_out(len(data), folds, pout, seed=seed, verbose=verbose)

    for fold_index, (train, test) in enumerate(folds):
        train_data, test_data = data[train], data[test]

        train_image_dir, val_image_dir, train_annot_dir, val_annot_dir = _setup_split_dirs(output_dir / f"{fold_index}")

        for image, annotation in tqdm(
                train_data,
                desc=f"Processing train {fold_index}",
                disable=not verbose
        ):
            shutil.copy(image, train_image_dir / image.name)
            shutil.copy(annotation, train_annot_dir / annotation.name)

        for image, annotation in tqdm(
                test_data,
                desc=f"Processing val {fold_index}",
                disable=not verbose
        ):
            shutil.copy(image, val_image_dir / image.name)
            shutil.copy(annotation, val_annot_dir / annotation.name)
