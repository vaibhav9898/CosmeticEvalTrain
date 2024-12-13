import numpy as np
import torch

def map_to_range(number):
    # Assuming the input range is 1 to 4
    min_value = 1
    max_value = 4

    # Perform min-max normalization
    normalized_value = (number - min_value) / (max_value - min_value)

    # Reverse the order to match the specified mapping (0 corresponds to 4)
    mapped_value = 1 - normalized_value

    return np.array(mapped_value, dtype=np.float32)

def reverse_mapping(mapped_value):
    # Assuming the input range is 1 to 4
    min_value = 1
    max_value = 4

    # Reverse the order to match the specified mapping (0 corresponds to 4)
    normalized_value = 1 - mapped_value

    # Reverse min-max normalization
    reversed_number = normalized_value * (max_value - min_value) + min_value

    # Round off to the nearest class
    reversed_number = torch.round(reversed_number)

    return reversed_number

def integer_to_one_hot(integer):
    if integer == 1:
        return [1, 0, 0, 0]
    elif integer == 2:
        return [0, 1, 0, 0]
    elif integer == 3:
        return [0, 0, 1, 0]
    elif integer == 4:
        return [0, 0, 0, 1]
    else:
        raise ValueError("Input integer must be 1, 2, 3, or 4.")

def kendall_tau(x, y):
    # Ensure the input tensors have the same size
    assert x.size(0) == y.size(0)

    n = x.size(0)

    # Calculate concordant and discordant pairs
    concordant, discordant = 0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1

    # Calculate Kendall's Tau
    tau = (concordant - discordant) / (0.5 * n * (n - 1))
    
    return tau

def calculate_mae(predicted_classes, actual_labels):
    # Ensure the input tensors have the same size
    # Calculate MAE
    mae = torch.mean(torch.abs(predicted_classes - actual_labels).float())
    return mae
