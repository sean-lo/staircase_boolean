import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import functools

from staircase.utils import (
    get_parity_fourier_fn,
    get_staircase_fourier_fn,
    get_complete_staircase_fourier_fn,
    get_complete_clipped_staircase_fourier_fn,
    get_msp_example_fourier_fn,
    get_half_msp_example_fourier_fn,
    get_random_staircase_fourier_fn,
    convert_fourier_fn_to_eval_fn,
)
from staircase.datasets import (
    generate_boolean_unbiased,
    eval_parity_fast,
    eval_staircase_fast,
    eval_complete_staircase_fast,
    eval_complete_clipped_staircase_fast,
    eval_msp_example_fast,
    eval_half_msp_example_fast,
)
from staircase.neural_net_architectures import ReLUResNet
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


def run_instance(
    data: pd.DataFrame,
    row_index: int,
    time_limit: float = 3600.0,
    save_fourier_fn: bool = True,
    save_losses: bool = True,
    save_records: bool = True,
    save_model: bool = False,
    write_log: bool = False,
):
    function_type = data.iloc[row_index].function_type
    degree = int(data.iloc[row_index].degree)
    depth = data.iloc[row_index].depth
    width = data.iloc[row_index].width
    num_variables = int(data.iloc[row_index].num_variables)
    num_samples = data.iloc[row_index].num_samples
    num_training_epochs = int(data.iloc[row_index].num_training_epochs)

    match function_type:
        case "parity":
            fourier_fn = get_parity_fourier_fn(degree)
            eval_fn = functools.partial(eval_parity_fast, d=degree)
        case "staircase":
            fourier_fn = get_staircase_fourier_fn(degree)
            eval_fn = functools.partial(eval_staircase_fast, d=degree)
        case "complete":
            fourier_fn = get_complete_staircase_fourier_fn(degree)
            eval_fn = functools.partial(eval_complete_staircase_fast, d=degree)
        case "clipped_complete":
            fourier_fn = get_complete_clipped_staircase_fourier_fn(degree)
            eval_fn = functools.partial(eval_complete_clipped_staircase_fast, d=degree)
        case "msp":
            fourier_fn = get_msp_example_fourier_fn(degree)
            eval_fn = functools.partial(eval_msp_example_fast, d=degree)
        case "half_msp":
            fourier_fn = get_half_msp_example_fourier_fn(degree)
            eval_fn = functools.partial(eval_half_msp_example_fast, d=degree)
        case "random":
            fourier_fn = get_random_staircase_fourier_fn(degree, 0)
            eval_fn = convert_fourier_fn_to_eval_fn(fourier_fn)
        case _:
            raise (NotImplementedError())
    if save_fourier_fn:
        pickle.dump(fourier_fn, open(f"fourier_fns/{row_index}.pkl", "wb"))

    match depth:
        case "two":
            num_layers = 2
        case "half_d":
            num_layers = degree // 2
        case "d":
            num_layers = degree
        case _:
            raise (NotImplementedError())

    match width:
        case "half_d":
            layer_width = degree // 2
        case "d":
            layer_width = degree
        case "2d":
            layer_width = degree * 2
        case "4d":
            layer_width = degree * 4
        case _:
            raise (NotImplementedError())

    match num_samples:
        case "1000d":
            erm_num_samples = 1000 * degree
        case _:
            raise (NotImplementedError())

    num_iter = num_training_epochs * erm_num_samples

    running_losses, running_pop_losses = run_train_eval_loop(
        n=num_variables,
        gen_fn=generate_boolean_unbiased,
        gen_fn_str="unbiased",
        eval_fn=eval_fn,
        eval_fourier_fn=fourier_fn,
        eval_fn_str="custom",
        erm=True,
        erm_num_samples=erm_num_samples,
        num_layers=num_layers,
        layer_width=layer_width,
        net_type=ReLUResNet,
        train_batch_size=20,
        num_iter=num_iter,
        learning_rate=0.01,
        learning_schedule=lr_sched,
        write_log=write_log,
        save_model=save_model,
    )
    # write losses to file
    if save_losses:
        np.array(running_losses).tofile(Path(f"losses/train_loss_{row_index}.npy"))
        np.array(running_pop_losses).tofile(Path(f"losses/val_losses_{row_index}.npy"))
    # write record to file
    if save_records:
        records = {
            "function_type": [function_type],
            "degree": [degree],
            "num_layers": [num_layers],
            "layer_width": [layer_width],
            "num_variables": [num_variables],
            "erm_num_samples": [erm_num_samples],
            "num_training_epochs": [num_training_epochs],
            "num_iter": [num_iter],
            "net_type": [ReLUResNet.__name__],
            "train_batch_size": [20],
            "learning_rate": [0.01],
            "train_loss_initial": [running_losses[0]],
            "val_loss_initial": [running_pop_losses[0]],
            "train_loss_final": [running_losses[-1]],
            "val_loss_final": [running_pop_losses[-1]],
        }
        pd.DataFrame.from_dict(records).to_csv(f"records/{row_index}.csv")
    return


def main():
    task_index = int(sys.argv[1])
    n_tasks = int(sys.argv[2])

    # testing_data = pd.read_csv("testing_data.csv")
    # for row_index in range(0, testing_data.shape[0], 1):
    #     run_instance(
    #         testing_data,
    #         row_index,
    #         time_limit=3600.0,
    #         save_fourier_fn=True,
    #         save_losses=True,
    #         save_records=True,
    #     )

    data = pd.read_csv("data.csv")

    print(f"Processing rows: {list(range(task_index, data.shape[0], n_tasks))}")
    for row_index in range(task_index, data.shape[0], n_tasks):
        print(row_index)
        run_instance(
            data, row_index, time_limit=3600.0,
            save_fourier_fn=True, save_losses=True, save_records=True,
        )
    return


if __name__ == "__main__":
    main()
