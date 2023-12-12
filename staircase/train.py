import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pickle
from pathlib import Path
import datetime

from staircase.neural_net_architectures import ReLUResNet

from staircase.utils import (
    eval_fourier_tuple,
    convert_fourier_fn_to_eval_fn,
    convert_fourier_fn_to_fourier_tuples,
)

from staircase.datasets import (
    BooleanDataset,
    ERMBooleanDataset,
)


def generate_dataloaders(
    n: int,
    gen_fn,
    eval_fn,
    erm: bool,
    erm_num_samples: int,
    batch_size: int,
):
    if erm:
        pop_dataset = BooleanDataset(n, gen_fn, eval_fn)
        pop_dataloader = DataLoader(pop_dataset, batch_size=batch_size, num_workers=0)
        dataset = ERMBooleanDataset(n, gen_fn, eval_fn, erm_num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        return pop_dataloader, dataloader
    else:
        dataset = BooleanDataset(n, gen_fn, eval_fn)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        return None, dataloader


def build_net(
    n: int,
    num_layers: int,
    layer_width: int,
    net_type=ReLUResNet,
):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")

    net = net_type(n, num_layers, layer_width)
    net.to(device)
    return net, device


def train_net_and_save_params(
    dataloader,
    pop_dataloader,
    criterion,
    net,
    device,
    num_iter: int,
    learning_rate: float,
    learning_schedule,
    erm: bool,
    erm_num_samples: int,
    run_dir: str | Path,
    save_model: bool = True,
):
    sgd_noise = 0
    print("Train_function started")
    net_dir = Path(run_dir) / "net"
    if save_model: net_dir.mkdir(exist_ok=True)

    dataloader_iter = iter(dataloader)
    if erm:
        pop_dataloader_iter = iter(pop_dataloader)

    running_losses = []
    running_loss = 0.0
    running_pop_losses = []
    running_pop_loss = 0.0
    print("Starting training")
    for iter_num in range(num_iter):  # loop over the dataset multiple times
        trainable_params = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.SGD(
            trainable_params,
            lr=learning_schedule(learning_rate, iter_num),
            momentum=0,
        )

        data = next(dataloader_iter)
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels[:, 0]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if erm:
            with torch.no_grad():
                pop_data = next(pop_dataloader_iter)
                pop_inputs, pop_labels = pop_data
                pop_inputs = pop_inputs.to(device)
                pop_labels = pop_labels.to(device)
                pop_labels = pop_labels[:, 0]
                pop_loss = criterion(net(pop_inputs), pop_labels)
                running_pop_loss += pop_loss.item()

        if iter_num % 100 == 0 and sgd_noise > 0:
            perturb_params = filter(lambda p: p.requires_grad, net.parameters())
            with torch.no_grad():
                for p in perturb_params:
                    noise_tensor = torch.randn(p.size()) * sgd_noise
                    noise_tensor = noise_tensor.to(device)
                    p.add_(noise_tensor)

        # print statistics
        running_loss += loss.item()
        if iter_num % erm_num_samples == erm_num_samples - 1:  # print every 1000 mini-batches
            if erm:
                print(
                    f"{iter_num:>6d}: Running train loss {running_loss / erm_num_samples:.6f}, Population loss {running_pop_loss / erm_num_samples:.6f}"
                )
                running_losses.append(running_loss)
                running_loss = 0.0
                running_pop_losses.append(running_pop_loss)
                running_pop_loss = 0.0
            else:
                print(f"{iter_num:>6d}: Running train loss {running_loss / erm_num_samples:.6f}")
                running_losses.append(running_loss)
                running_loss = 0.0

        if save_model and iter_num % erm_num_samples == 0:
            net_path = net_dir / f"{iter_num}.pkl"
            pickle.dump(net, open(net_path, "wb"))
            print(f"{iter_num:>6d}: Saved network parameters.")

    return running_losses, running_pop_losses


def output_losses_and_fourier_coeffs(
    device,
    dataloader,
    iter_range,
    criterion,
    track_fourier_coeffs_tuples: list,
    run_dir: str | Path,
):
    print("Loss_output function started")
    dataloader_iter = iter(dataloader)

    with torch.no_grad():
        running_losses = []
        running_coeffs = []
        for iter_num in iter_range:
            # print(iter_num)
            net_path = Path(run_dir) / f"net/{iter_num}.pkl"
            net = pickle.load(open(net_path, "rb"))
            net.to(device)

            # print("Loading data")
            total_loss = 0
            for inner_iter_num in range(1):  # loop over the dataset multiple times
                data = next(dataloader_iter)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels[:, 0]

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                running_coeffs.append([])
                for fourier_ind, fourier_tuple in enumerate(
                    track_fourier_coeffs_tuples
                ):
                    # fourier_tuple_labels is a tensor of size (batch_size,)
                    # each entry is the monomial (prescribed by fourier_tuple) evaluated at that input
                    fourier_tuple_labels = eval_fourier_tuple(inputs, fourier_tuple)
                    # fourier_tuple_val
                    fourier_tuple_val = torch.mean(outputs * fourier_tuple_labels)
                    running_coeffs[-1].append(fourier_tuple_val)

                total_loss += loss.item()

                print(
                    f"{iter_num:>6d}: Eval loss {total_loss:.6f}, \nCoeffs {running_coeffs[-1]}"
                )
            running_losses.append(total_loss)

        print(running_losses)
        pickle.dump(
            (iter_range, running_losses),
            open(Path(run_dir) / "losses.pkl", "wb"),
        )
        pickle.dump(
            (iter_range, running_coeffs),
            open(Path(run_dir) / "coeffs.pkl", "wb"),
        )

        return running_losses, running_coeffs


def run_train_eval_loop(
    n: int,
    # dataset params
    gen_fn,
    gen_fn_str: str,
    eval_fourier_fn: dict,
    eval_fn_str: str,
    erm: bool,
    erm_num_samples: int,
    # architecture params
    num_layers: int,
    layer_width: int,
    net_type,
    # training params
    train_batch_size: int,
    num_iter: int,
    # num_epochs: int,
    learning_rate: float,
    learning_schedule,
    # eval params
    # eval_batch_size: int,
    # iter_range,
    criterion=nn.MSELoss(),
    # optional: eval_fn
    eval_fn=None,
    write_log: bool = True,
    save_model: bool = True,
):
    if eval_fn is None:
        eval_fn = convert_fourier_fn_to_eval_fn(eval_fourier_fn)
    track_fourier_coeffs_tuples = convert_fourier_fn_to_fourier_tuples(
        eval_fourier_fn,
        n,
    )

    # Initialize run directory
    time_str = datetime.datetime.now().isoformat(timespec="seconds")
    run_dir = f"../trained_wts/{gen_fn_str}_{eval_fn_str}/{time_str}/"
    if write_log:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "readme.txt").write_text(
            f"""time:               {time_str}
    gen_fn:             {gen_fn_str}
    eval_fn:            {eval_fn_str}
    n:                  {n}
    num_layers:         {num_layers}
    layer_width:        {layer_width}
    net_type:           {net_type.__name__}
    train_batch_size:   {train_batch_size}
    num_iter:           {num_iter}
    learning_rate:      {learning_rate}
    erm:                {erm}
    erm_num_samples:    {erm_num_samples}
            """
        )
        pickle.dump(eval_fourier_fn, open(Path(run_dir) / "fourier_fn.pkl", "wb"))

    pop_dataloader, dataloader = generate_dataloaders(
        n=n,
        gen_fn=gen_fn,
        eval_fn=eval_fn,
        erm=erm,
        erm_num_samples=erm_num_samples,
        batch_size=train_batch_size,
    )
    net, device = build_net(
        n=n,
        num_layers=num_layers,
        layer_width=layer_width,
        net_type=net_type,
    )
    running_losses, running_pop_losses = train_net_and_save_params(
        dataloader=dataloader,
        pop_dataloader=pop_dataloader,
        criterion=criterion,
        net=net,
        device=device,
        num_iter=num_iter,
        erm_num_samples=erm_num_samples,
        learning_rate=learning_rate,
        learning_schedule=learning_schedule,
        erm=erm,
        run_dir=run_dir,
        save_model=save_model,
    )
    # _, eval_dataloader = generate_dataloaders(
    #     n=n,
    #     gen_fn=gen_fn,
    #     eval_fn=eval_fn,
    #     erm=False,
    #     erm_num_samples=erm_num_samples,
    #     batch_size=eval_batch_size,
    # )
    # losses, fourier_coeffs = output_losses_and_fourier_coeffs(
    #     device=device,
    #     dataloader=eval_dataloader,
    #     iter_range=iter_range,
    #     criterion=criterion,
    #     track_fourier_coeffs_tuples=track_fourier_coeffs_tuples,
    #     run_dir=run_dir,
    # )
    return running_losses, running_pop_losses
