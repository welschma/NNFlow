from __future__ import absolute_import, division, print_function
import numpy as np
import sys

class DataFrame:
    """TODO
    """

    def __init__(self, array):
        self.x = array[:, 1:-1]
        self.y = array[:, :1]
        self.w = array[:, -1:]
        self.n = self.x.shape[0]
        self.nvariables = self.x.shape[1]
        self.next_id=0
        self.shuffle()
        self._check_std()

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

    def _check_std(self):
        data_std = np.std(self.x, axis=0)
        if np.count_nonzero(data_std==0.0):
            print('Std of training data:')
            print(data_std)
            sys.exit('Remove Variable with std 0.0') 

