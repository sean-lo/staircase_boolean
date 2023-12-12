import torch
import numpy as np
import itertools
import random


def convert_fourier_fn_to_eval_fn(
    fourier_fn: dict,
):
    def eval_fn(
        x: torch.Tensor,
    ):
        return torch.FloatTensor(
            [
                sum(
                    fourier_coeff * torch.prod(x[list(fourier_tuple)])
                    for fourier_tuple, fourier_coeff in fourier_fn.items()
                )
            ]
        )

    return eval_fn


def convert_fourier_fn_to_fourier_tuples(
    fourier_fn: dict,
    n: int,
):
    track_fourier_coeffs_tuples = [tuple(np.ones((n,)))]
    for fourier_tuple in fourier_fn.keys():
        curr_coeff = np.ones((n,))
        curr_coeff[list(fourier_tuple)] = -1
        track_fourier_coeffs_tuples.append(tuple(curr_coeff))
    return track_fourier_coeffs_tuples


def get_staircase_fourier_fn(d: int):
    fourier_fn = {}
    for i in range(d):
        fourier_fn[tuple(range(i + 1))] = 1
    return fourier_fn


def get_parity_fourier_fn(d: int):
    fourier_fn = {}
    fourier_fn[tuple(range(d))] = 1
    return fourier_fn


def get_complete_staircase_fourier_fn(d: int):
    fourier_fn = {}
    for s in range(1, d + 1):
        for S in itertools.combinations(range(d), s):
            fourier_fn[S] = 1
    return fourier_fn


def get_complete_clipped_staircase_fourier_fn(d: int):
    fourier_fn = {}
    for s in range(1, d):
        for S in itertools.combinations(range(d), s):
            fourier_fn[S] = 1
    return fourier_fn


def get_msp_example_fourier_fn(d: int):
    fourier_fn = {}
    for i in range(d):
        fourier_fn[(i,)] = 1
    fourier_fn[tuple(range(d))] = 1
    return fourier_fn


def get_half_msp_example_fourier_fn(d: int):
    fourier_fn = {}
    for i in range(d):
        fourier_fn[(i,)] = 1
    for i in range(d // 2 + 1):
        fourier_fn[tuple(range(i, d // 2 + i))] = 1
    fourier_fn[tuple(range(d))] = 1
    return fourier_fn


def get_random_staircase_fourier_fn(d: int, seed: int):
    random.seed(seed)
    return _get_random_staircase_fourier_fn(0, d)


def _get_random_staircase_fourier_fn(
    i_start: int,
    i_end: int,
):
    # gets a staircase function with indices in [i_start, i_end)
    if i_end - i_start == 1:
        return {(i_start,): 1}
    elif i_end - i_start == 2:
        if random.random() > 0.5:
            # x_0 + x_0 x_1
            return {
                (i_start,): 1,
                (
                    i_start,
                    i_start + 1,
                ): 1,
            }
        else:
            # x_0 + x_1
            return {
                (i_start,): 1,
                (i_start + 1,): 1,
            }
    else:
        i_mid = (i_end + i_start) // 2
        fourier_fn_0 = _get_random_staircase_fourier_fn(i_start, i_mid)
        fourier_fn_1 = _get_random_staircase_fourier_fn(i_mid, i_end)
        if random.random() > 0.5:
            # x_0 + x_0 x_1
            fourier_fn = fourier_fn_0.copy()
            for k1, k2 in itertools.product(fourier_fn_0.keys(), fourier_fn_1.keys()):
                fourier_fn[k1 + k2] = 1
            return fourier_fn
        else:
            # x_0 + x_1
            return fourier_fn_0 | fourier_fn_1


def get_multi_staircase_fourier_fn(d_1: int, d_2: int):
    fourier_fn = {}
    for i in range(d_1):
        fourier_fn[tuple(range(i + 1))] = 1
    for i in range(d_2 - 1):
        fourier_fn[tuple([0] + list(range(d_1, d_1 + i + 1)))] = 1
    return fourier_fn


def eval_fourier_tuple(
    inputs: torch.Tensor,
    fourier_tuple: tuple,
):
    """
    inputs: tensor of size (batch_size, n)
    fourier_tuple: tuple of length n, with entries in {-1, 1}

    returns labels, tensor of size (batch_size,)
        each entry is the monomial evaluated at that input
    """
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
