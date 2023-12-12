import torch

import numpy as np
import random

from staircase.neural_net_architectures import ReLUResNet

from staircase.datasets import (
    generate_boolean_unbiased,
)
from staircase.train import run_train_eval_loop


def lr_sched(
    learning_rate: float,
    iter_num: int,
):
    lr_decay = 15000  # learning rate decay parameter
    if iter_num < lr_decay:
        return learning_rate
    if iter_num < 2 * lr_decay:
        return learning_rate / 2
    if iter_num < 3 * lr_decay:
        return learning_rate / 4
    if iter_num < 4 * lr_decay:
        return learning_rate / 8
    if iter_num < 5 * lr_decay:
        return learning_rate / 16
    if iter_num < 6 * lr_decay:
        return learning_rate / 32
    else:
        return learning_rate / 64
    # return learning_rate*math.exp(-iter_num/lr_decay)


if __name__ == "__main__":
    torch.random.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    learning_rate = 0.01
    learning_schedule = lr_sched
    train_batch_size = 20
    eval_batch_size = 30000
    erm = True
    erm_num_samples = 60000
    num_iter = erm_num_samples
    net_type = ReLUResNet
    iter_range = range(0, num_iter, 1000)

    n = 4
    num_layers = 1
    layer_width = 4

    fourier_fn = {
        (1,): 1,
        (2,): 1,
        (0, 1): 1,
        (0, 2): 1,
        (0, 1, 3): 1,
    }
    run_train_eval_loop(
        n=n,
        gen_fn=generate_boolean_unbiased,
        gen_fn_str="unbiased",
        eval_fourier_fn=fourier_fn,
        eval_fn_str="custom",
        erm=erm,
        erm_num_samples=erm_num_samples,
        num_layers=num_layers,
        layer_width=layer_width,
        net_type=net_type,
        train_batch_size=train_batch_size,
        num_iter=num_iter,
        learning_rate=learning_rate,
        learning_schedule=learning_schedule,
        # eval_batch_size=eval_batch_size,
        # iter_range=iter_range,
    )
