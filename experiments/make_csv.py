import pandas as pd
import itertools

data = {
    "function_type": [],
    "degree": [],
    "depth": [],
    "width": [],
    "num_variables": [],
    "num_samples": [],
    "num_training_epochs": [],
}
for (
    function_type,
    degree,
    depth,
    width,
    num_variables,
    num_samples,
    num_training_epochs,
) in itertools.product(
    [
        "parity",
        "staircase",
        "complete",
        "clipped_complete",
        "msp",
        "half_msp",
        "random",
    ],
    [8, 12, 16],
    ["two", "half_d", "d"],
    ["half_d", "d", "2d", "4d"],
    [32],
    ["1000d"],
    [100],
):
    data["function_type"].append(function_type)
    data["degree"].append(degree)
    data["depth"].append(depth)
    data["width"].append(width)
    data["num_variables"].append(num_variables)
    data["num_samples"].append(num_samples)
    data["num_training_epochs"].append(num_training_epochs)

pd.DataFrame(data).to_csv("data.csv")


testing_data = {
    "function_type": [],
    "degree": [],
    "depth": [],
    "width": [],
    "num_variables": [],
    "num_samples": [],
    "num_training_epochs": [],
}
for (
    function_type,
    degree,
    depth,
    width,
    num_variables,
    num_samples,
    num_training_epochs,
) in itertools.product(
    [
        "parity",
        "staircase",
        "complete",
        "clipped_complete",
        "msp",
        "half_msp",
        "random",
    ],
    [8],
    ["two"],
    ["half_d"],
    [32],
    ["1000d"],
    [100],
):
    testing_data["function_type"].append(function_type)
    testing_data["degree"].append(degree)
    testing_data["depth"].append(depth)
    testing_data["width"].append(width)
    testing_data["num_variables"].append(num_variables)
    testing_data["num_samples"].append(num_samples)
    testing_data["num_training_epochs"].append(num_training_epochs)

pd.DataFrame(testing_data).to_csv("testing_data.csv")
