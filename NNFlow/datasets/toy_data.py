"""Creates a 2d toy dataset that can be used to validate changes in the
classifier.
"""

import numpy as np
from ..data_frame import DataFrame

def get_bkg(N):
    """Returns N background events. A label (0.0) and a weight (1.0) ist added.
    The array matches the required form of the DataFrame.

    Arguments:
    ------------
    N (int) :
    Number of events.
    """
    # create N events on a circular arc (<- maybe not the right word for it)
    r = np.random.normal(loc=3., scale=0.8, size=N)
    theta = np.random.uniform(low=1.0, high=5.0, size=N)
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    # add labels and weights
    label = np.full((N,1), 0.0)
    weight = np.full((N,1), 1.0)
    bkg = np.hstack((label,x.reshape(-1,1),y.reshape(-1,1),weight))
    
    return bkg

def get_sig(N):
    """Returns N signal events following a 2d gaussian distribution. A label 
    (0.0) and a weight (1.0) ist added.The array matches the required form of
    the DataFrame.
    """
    # create N events following a simple 2d gaussian distribution
    mean = [0, 0]
    cov = [[3, 0], [0, 3]]  
    x, y = np.random.multivariate_normal(mean, cov, N).T

    # add labels and weights
    label = np.full((N,1), 1.0)
    weight = np.full((N,1), 1.0)
    sig = np.hstack((label,x.reshape(-1,1),y.reshape(-1,1),weight))

    return sig

def load_data():
    """Loads training, validation datasets. 1e6 events are split in ratio 1:5 for
    validation:training.

    Returns:
    -----------
    train (DataSet) :
    800k training events.
    val (DataSet) :
    200k valdidation events
    """
    N = 1000000
    sig = get_sig(N)
    bkg = get_bkg(N)

    #split data
    split = int(0.2*N)
    train = np.vstack((sig[:-split], bkg[:-split]))
    val = np.vstack((sig[-split:], bkg[-split:]))
    
    train = DataFrame(train)
    val = DataFrame(val)

    return train, val
