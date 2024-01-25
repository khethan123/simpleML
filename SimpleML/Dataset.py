import numpy as np

# Sine dataset
def sine_data(samples=1000):
    """
    Generate sine dataset.

    Args:
        samples (int): Number of samples. Defaults to 1000.

    Returns:
        tuple: X and y arrays.
    """
    x = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * x).reshape(-1, 1)
    return x, y


# Cosine dataset
def cosine_data(samples=1000):
    """
    Generate cosine dataset.

    Args:
        samples (int): Number of samples. Defaults to 1000.

    Returns:
        tuple: X and y arrays.
    """
    x = np.arange(samples).reshape(-1, 1) / samples
    y = np.cos(2 * np.pi * x).reshape(-1, 1)
    return x, y


# Tan dataset
def tan_data(samples=1000):
    """
    Generate tangent dataset.

    Args:
        samples (int): Number of samples. Defaults to 1000.

    Returns:
        tuple: X and y arrays.
    """
    x = np.arange(samples).reshape(-1, 1) / samples
    y = np.tan(2 * np.pi * x**0.5).reshape(-1, 1)  # change the power of x
    return x, y


# Log dataset
def log_data(samples=1000):
    """
    Generate log dataset.

    Args:
        samples (int): Number of samples. Defaults to 1000.

    Returns:
        tuple: X and y arrays.
    """
    x = np.arange(1, samples + 1).reshape(-1, 1) / samples
    y = np.log(x).reshape(-1, 1)
    # OR
    # X = np.random.exponential(scale=1, size=1000).reshape(-1, 1) / 1000
    # y = np.log(X).reshape(-1, 1)
    return x, y


def spiral_data(samples, classes):
    """
    Generate spiral dataset.

    Args:
        samples (int): Number of samples.
        classes (int): Number of classes.

    Returns:
        tuple: X and y arrays.
    """
    x = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return x, y


def vertical_data(samples, classes):
    """
    Generate vertical dataset.

    Args:
        samples (int): Number of samples.
        classes (int): Number of classes.

    Returns:
        tuple: X and y arrays.
    """
    x = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        x[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    return x, y