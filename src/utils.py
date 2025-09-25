# preprocessing using normalising
def z_normalize(data):
    # Mean and standard deviation calculated over the sample and time dimensions, for each channel
    means = data.mean(axis=(0, 1), keepdims=True)  # Shape (1, 1, 34)
    stds = data.std(axis=(0, 1), keepdims=True)  # Shape (1, 1, 34)

    # Avoid division by zero in case of zero variance
    stds[stds == 0] = 1

    # Standardize the data
    normalized_data = (data - means) / stds

    return normalized_data