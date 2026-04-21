import numpy as np

def offsets_to_absolute(sample):
    """
    Converts the relative representation stroke data back to absolute coordinates
    """
    
    abs_sample = sample.copy()
    abs_sample[:, 0] = np.cumsum(sample[:, 0])  # x
    abs_sample[:, 1] = np.cumsum(sample[:, 1])  # y
    # column 2 (pen state) stays unchanged
    return abs_sample