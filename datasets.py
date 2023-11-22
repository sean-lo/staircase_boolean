import torch
import math
import functools

from torch.utils.data import IterableDataset, DataLoader


def generate_boolean_unbiased(n: int):
    """
    Generates a random tensor of shape (n,) with entries in {-1, 1}.
    """
    return torch.FloatTensor(torch.bernoulli(torch.full((n,), 0.5)) * 2 - 1)


def generate_boolean_biased(n: int, p: float):
    return (torch.bernoulli(torch.full((n,), p)) - p) / torch.sqrt(torch.mul(p, 1 - p))


def generate_gaussian(n: int):
    return torch.normal(0, 1, size=(n,))


def eval_parity_fast(x: torch.Tensor, d: int):
    return torch.FloatTensor([torch.cumprod(x, 0)[d - 1]])


def eval_staircase_fast(x: torch.Tensor, d: int):
    return torch.FloatTensor(
        [torch.cumsum(torch.cumprod(x, 0), 0)[d - 1] / math.sqrt(d)]
    )


def eval_multi_stair_fast(x: torch.Tensor, d_1: int, d_2: int):
    return (
        eval_staircase_fast(x, d_1)
        + eval_staircase_fast(torch.cat((x[0:1], x[d_1:])), d_2)
        - x[0]
    )


# Evaluate function given by Fourier representation (i.e., multilinear polynomial) at a certain point:
# Multilinear polynomial is given as a tuple : float dict, where each tuple consists of the relevant indices.
def eval_fourier_fn(x, fourier_fn):
    return torch.FloatTensor(
        [
            torch.sum(
                torch.Tensor(
                    [
                        coeff * torch.prod(x[list(term)])
                        for term, coeff in fourier_fn.items()
                    ]
                )
            )
        ]
    )


class BooleanDataset(IterableDataset):
    def __init__(self, n: int, gen_fn, eval_fn):
        """
        Args:
            n (int): input length.
            gen_fn (fn): generation function generating $x \in \{-1, 1\}^n$
            eval_fn (fn): evaluation function $\{-1,1\}^n \to \RR$
        """
        self.n = n
        self.gen_fn = gen_fn
        self.eval_fn = eval_fn

    def __iter__(self):
        while True:
            x = self.gen_fn(self.n)
            y = self.eval_fn(x)
            yield x, y


class ERMBooleanDataset(IterableDataset):
    def __init__(
        self,
        n: int,
        gen_fn,
        eval_fn,
        erm_num_samples: int,
    ):
        self.n = n
        self.gen_fn = gen_fn
        self.eval_fn = eval_fn
        self.erm_num_samples = erm_num_samples

        self.counter = 0

        fn_dataset = BooleanDataset(n, gen_fn, eval_fn)
        dataloader = DataLoader(fn_dataset, batch_size=erm_num_samples, num_workers=0)
        self.xs, self.ys = next(iter(dataloader))
        return

    def __iter__(self):
        while True:
            self.counter += 1
            if self.counter == self.erm_num_samples:
                self.counter = 0
            yield self.xs[self.counter, :], self.ys[self.counter, :]
