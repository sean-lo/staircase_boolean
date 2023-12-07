import torch

import functools
import numpy as np
import random

from staircase.neural_net_architectures import ReLUResNet
from staircase.utils import (
    get_multi_staircase_fourier_fn,
)

from staircase.datasets import (
    generate_boolean_unbiased,
    eval_multi_stair_fast,
)
from staircase.train import run_train_eval_loop


def lr_sched_multi(
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
    d_1: int,
    d_2: int,
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
    for eval_fn, eval_fn_str, eval_fourier_fn in zip(
        [
            functools.partial(eval_multi_stair_fast, d_1=d_1, d_2=d_2),
        ],
        [
            "stair_multi",
        ],
        [
            get_multi_staircase_fourier_fn(d_1, d_2),
        ],
    ):
        run_train_eval_loop(
            n=n,
            gen_fn=generate_boolean_unbiased,
            gen_fn_str="unbiased",
            eval_fourier_fn=eval_fourier_fn,
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
            eval_batch_size=eval_batch_size,
            iter_range=iter_range,
        )


if __name__ == "__main__":
    torch.random.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    num_iter = 50000
    refresh_save_rate = 1000
    learning_rate = 0.004
    learning_schedule = lr_sched_multi
    train_batch_size = 20
    eval_batch_size = 30000
    erm = True
    erm_num_samples = 100000
    net_type = ReLUResNet
    iter_range = range(0, num_iter, 1000)

    n = 30
    d_1 = 7  # degree of staircase
    d_2 = 7
    num_layers = 5
    layer_width = 50

    main(
        n=n,
        d_1=d_1,
        d_2=d_2,
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
        eval_batch_size=eval_batch_size,
        iter_range=iter_range,
    )
