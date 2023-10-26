import numpy as np

def generate_test_data(
    data_dim: int,
    sequence_len: int,
    n_samples: int,
    drift_step_size: float=0.001,
    mode: str='l2'
) -> np.ndarray:
    """Generate test data. Each row is produced by adding noise to the previous row.

    Args:
        data_dim (int): Data dimension
        sequence_len (int): Length of data to generate
        n_samples (int): Number of data points to generate
        drift_step_size (float, optional): Amount of data drift between rows. Defaults to 0.001.
        mode (str, optional): Operations to perform on generated data. Currently supports 'l2' and 'clip' 

    Returns:
        np.ndarray: Generated test data
    """
    data = []

    for i in range(n_samples):
        x = np.zeros((sequence_len, data_dim))
        drift_vector = np.random.normal(size=(1, data_dim))

        x[0] = np.random.normal(size=(1, data_dim))
        if mode == 'l2':
            # normalize l2 length to 1.0
            x[0] = x[0] / np.linalg.norm(x[0])
        elif mode == 'clip':
            x[0] = np.clip(x[0], -1, 1)

        for i in range(sequence_len - 1):
            x[i+1] = x[i] + drift_step_size * drift_vector
            if mode == 'l2':
                # normalize l2 length to 1.0
                x[i+1] = x[i+1] / np.linalg.norm(x[i+1])
            elif mode == 'clip':
                x[i+1] = np.clip(x[i+1], -1, 1)

        data.append(x)

    data = np.array(data).astype(np.float32)

    return data

if __name__ == '__main__':
    test_data = generate_test_data(n_samples=1, data_dim=768, sequence_len=77)
    print(f'Shape: {test_data.shape}')
    print(test_data[0, :5, :5])