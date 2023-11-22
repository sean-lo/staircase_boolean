import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim


def get_staircase_fourier_fn(n: int):
    fourier_fn = {}
    for i in range(n):
        fourier_fn[tuple(range(i + 1))] = 1
    return fourier_fn


def get_sparse_fourier_fn(n: int):
    fourier_fn = {}
    fourier_fn[tuple(range(n))] = 1
    return fourier_fn


def truthtable(n: int, eval_fn):
    return [eval_fn(x) for x in itertools.product([1, -1], repeat=n)]


def is_stair_coeff(tup):
    isOne = False
    for j in tup:
        if j == 1:
            isOne = True
        else:
            if isOne:
                return False
    return True


def plot_fourier_coeffs(stair_fourier_coeffs, title):
    if stair_fourier_coeffs:
        stair_fourier_coeffs = np.asarray(stair_fourier_coeffs).T
        print(stair_fourier_coeffs.shape)
        plt.figure()
        for i in range(1, 20):
            plt.plot(
                range(stair_fourier_coeffs.shape[1]),
                stair_fourier_coeffs[i, :],
                label=str(i),
            )
        plt.legend()
        plt.title(title)
        plt.show()


def eval_fourier_tuple(inputs, fourier_tuple):
    # print(inputs,fourier_tuple)
    labels = torch.ones(len(inputs[:, 1]))
    for j, v in enumerate(fourier_tuple):
        if v == -1:
            labels = labels * inputs[:, j]
    return labels


def get_staircase_fourier_coeff_tuples(n: int, d: int):
    track_fourier_coeffs_tuples = []
    for j in range(d + 1):
        curr_coeff = []
        for i in range(n):
            if i < j:
                curr_coeff.append(-1)
            else:
                curr_coeff.append(1)
        curr_coeff = tuple(curr_coeff)
        track_fourier_coeffs_tuples.append(curr_coeff)
    return track_fourier_coeffs_tuples


def get_multi_staircase_fourier_coeff_tuples(n: int, d_1: int, d_2: int):
    track_fourier_coeffs_tuples = []
    for j in range(d_1 + 1):
        curr_coeff = []
        for i in range(n):
            if i < j:
                curr_coeff.append(-1)
            else:
                curr_coeff.append(1)
        curr_coeff = tuple(curr_coeff)
        track_fourier_coeffs_tuples.append(curr_coeff)
    for j in range(d_1 + d_2):
        curr_coeff = []
        if j <= d_1:
            continue
        for i in range(n):
            if i == 0:
                curr_coeff.append(-1)
            if (i < j - 1) and (i >= d_1 - 1):
                curr_coeff.append(-1)
            else:
                curr_coeff.append(1)
        track_fourier_coeffs_tuples.append(tuple(curr_coeff))
    return track_fourier_coeffs_tuples
