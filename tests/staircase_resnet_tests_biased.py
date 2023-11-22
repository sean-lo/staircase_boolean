import torch

import functools
import numpy as np
import random

from neural_net_architectures import ReLUResNet
from utils import get_staircase_fourier_coeff_tuples

from datasets import (
    generate_boolean_biased,
    eval_parity_fast,
    eval_staircase_fast,
)
from train import run_train_eval_loop


def lr_sched_biased(
    learning_rate: float,
    iter_num: int,
):
    lr_decay = 15000  # learning rate decay parameter
    if iter_num < lr_decay:
        return learning_rate
    if iter_num < 2 * lr_decay:
        return learning_rate / 2
    if iter_num < 3 * lr_decay:
        return learning_rate / 2
    if iter_num < 4 * lr_decay:
        return learning_rate / 4
    if iter_num < 5 * lr_decay:
        return learning_rate / 4
    if iter_num < 6 * lr_decay:
        return learning_rate / 8
    else:
        return learning_rate / 16
    # return learning_rate*math.exp(-iter_num/lr_decay)


def main(
    n: int,
    p: float,
    d: int,
    erm: bool,
    erm_num_samples: int,
    num_layers: int,
    layer_width: int,
    net_type,
    train_batch_size: int,
    num_iter: int,
    learning_rate: float,
    learning_schedule,
    refresh_save_rate: int,
    eval_batch_size: int,
    iter_range,
):
    track_fourier_coeffs_tuples = get_staircase_fourier_coeff_tuples(n, d)
    for eval_fn, eval_fn_str in zip(
        [
            functools.partial(eval_staircase_fast, d=d),
            functools.partial(eval_parity_fast, d=d),
        ],
        [
            "stair",
            "parity",
        ],
    ):
        run_train_eval_loop(
            n=n,
            gen_fn=generate_boolean_biased,
            gen_fn_str="biased",
            eval_fn=eval_fn,
            eval_fn_str=eval_fn_str,
            erm=erm,
            erm_num_samples=erm_num_samples,
            num_layers=num_layers,
            layer_width=layer_width,
            net_type=net_type,
            train_batch_size=train_batch_size,
            num_iter=num_iter,
            learning_rate=learning_rate,
            learning_schedule=learning_schedule,
            refresh_save_rate=refresh_save_rate,
            track_fourier_coeffs_tuples=track_fourier_coeffs_tuples,
            eval_batch_size=eval_batch_size,
            iter_range=iter_range,
        )


if __name__ == "__main__":
    torch.random.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    n = 30
    p = 0.75
    d = 7
    num_layers = 5
    layer_width = 40
    num_iter = 100000
    refresh_save_rate = 1000
    learning_rate = 0.01
    learning_schedule = lr_sched_biased
    train_batch_size = 20
    eval_batch_size = 30000
    erm = True
    erm_num_samples = 60000
    net_type = ReLUResNet
    iter_range = range(0, num_iter, 1000)

    main(
        n=n,
        p=p,
        d=d,
        num_layers=num_layers,
        layer_width=layer_width,
        num_iter=num_iter,
        refresh_save_rate=refresh_save_rate,
        learning_rate=learning_rate,
        learning_schedule=learning_schedule,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        erm=erm,
        erm_num_samples=erm_num_samples,
        net_type=net_type,
        iter_range=iter_range,
    )
