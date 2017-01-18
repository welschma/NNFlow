import numpy as np

class DataFrame:
    """TODO
    """

    def __init__(self, array):
        self.x = array[:, 1:-1]
        self.y = array[:, :1]
        self.w = array[:, -1:]
        self.n = self.x.shape[0]
        self.nfeatures = self.x.shape[1]
        self.shuffle()

    def shuffle(self):
        perm = np.random.permutation(self.n)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.w = self.w[perm]

        self.next_id = 0

    def next_batch(self, batch_size):
        if self.next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self.next_id
        self.next_id += batch_size

        return (self.x[cur_id:cur_id+batch_size],
                self.y[cur_id:cur_id+batch_size],
                self.w[cur_id:cur_id+batch_size])

def load_data(even_path, odd_path):
    """Load even and odd numpy arrays and return train, val and test
    DataFrames.
    """
    train = np.load(even_path)
    odd = np.load(odd_path)

    # split odd in valdiation and test array, ratio 1:9
    # ratio between signal and background is kept
    bins, _ = np.histogram(odd[:, 0])
    n_sig, n_bg = bins[-1], bins[0]
    n_events = odd.shape[0]
    
    val_sig = odd[:int(0.1*n_sig)]
    val_bg = odd[n_events - int(0.1*n_bg):]
    
    test = odd[int(0.1*n_sig):n_events-int(0.1*n_bg)]
    val = np.vstack((val_sig, val_bg))
    
    return DataFrame(train), DataFrame(val), DataFrame(test)
