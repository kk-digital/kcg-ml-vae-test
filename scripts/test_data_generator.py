import numpy as np

def generate_test_data(
    data_dim: int,
    sequence_len: int,
    drift_step_size: float=0.001
) -> np.ndarray:
    """Generate test data. Each row is produced by adding noise to the previous row

    Args:
        data_dim (int): Data dimension
        sequence_len (int): Length of data to generate
        drift_step_size (float, optional): Amount of data drift between rows. Defaults to 0.001.

    Returns:
        np.ndarray: Generated test data
    """
    x = np.zeros((sequence_len, data_dim))
    drift_vector = np.random.normal(size=(1, data_dim))

    x[0] = np.random.normal(size=(1, data_dim))
    # normalize l2 length to 1.0
    x[0] = x[0] / np.linalg.norm(x[0])
    for i in range(sequence_len - 1):
        x[i+1] = x[i] + drift_step_size * drift_vector
        # normalize l2 length to 1.0
        x[i+1] = x[i+1] / np.linalg.norm(x[i+1])

    return x

if __name__ == '__main__':
    test_data = generate_test_data(data_dim=768, sequence_len=77)
    print(f'Shape: {test_data.shape}')
    print(test_data[:5, :5])