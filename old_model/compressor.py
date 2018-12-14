from __future__ import print_function
from __future__ import division
import multiprocessing
import time

import numpy as np
from sklearn.linear_model import Lasso
import random


class Compressor(object):
    def __init__(self, bases, max_non_zero_entry=10):
        m, n = bases.shape
        self.bases = np.zeros((m+1, n)) # each column is a base
        self.bases[:-1, :] = bases
        self.alpha_t, self.beta_t = 1.0, 2.0
        self.max_non_zero_entry = max_non_zero_entry

    def loss_fit(self, x, y):
        diff = self.bases.dot(x) - y
        return np.sum(diff * diff)

    def loss_r1(self, x):
        return np.sum(np.abs(x))

    def loss_r2(self, x): # L2 loss
        return (np.sum(x) - 1) ** 2

    def loss(self, x, y):
        lf_l2, l1, l2 = self.loss_fit(x, y), self.loss_r1(x), self.loss_r2(x)
        return lf_l2 - l2, l1, l2

    def show(self):
        print("Base shape:", self.bases.shape)
        print("alpha_t = {0}, beta_t = {1}".format(self.alpha_t, self.beta_t))

    def fit(self, y, fill_to_max=True, verbose=False):
        """fill_to_max: whether or not to return `max_non_zero_entry` items. 
        i.e.: whether to include tailing zeros if # of non zero entries is less than `max_non_zero_entry`. """
        y = np.r_[y, 1]
        alpha_coefficient = 2.0 * self.bases.shape[0]
        min_alpha, max_alpha = 1e-4, 1000
        self.alpha_t = 15.0

        shrinkage_factor, expand_factor = 0.9, 1.5
        clf = Lasso(alpha=self.alpha_t / alpha_coefficient)

        # Check if max_alpha is large enough (for debug purpose)
        # clf.alpha = max_alpha / alpha_coefficient
        # clf.fit(self.bases, y)
        # x = clf.coef_
        # num_non_zero_entry = np.count_nonzero(x)
        # if num_non_zero_entry > self.max_non_zero_entry:
        #     print("max_alpha = {0}, clf.alpha = {1} too small!".format(max_alpha, clf.alpha))


        while True:
            clf.alpha = self.alpha_t / alpha_coefficient
            clf.fit(self.bases, y)
            x = clf.coef_
            num_non_zero_entry = np.count_nonzero(x)
            if num_non_zero_entry > self.max_non_zero_entry:
                # too many non zero entries => current alpha too small
                min_alpha = self.alpha_t
                new_alpha_t = self.alpha_t * expand_factor
                if new_alpha_t > max_alpha:
                    self.alpha_t = (new_alpha_t + max_alpha) / 2.0
                else:
                    self.alpha_t = new_alpha_t
            else:
                # too few non zero entries => current alpha too large
                max_alpha = self.alpha_t
                self.alpha_t *= shrinkage_factor
                if self.alpha_t < min_alpha:
                    break
        if verbose:
            lf, l1, l2 = self.loss(x, y)
            print(lf, l1, l2)
            print("Alpha range: {0} {1}".format(min_alpha, max_alpha))
            print("# of non-zero entries: {0}".format(np.count_nonzero(x)))


        # indices and values are all 1-D np.ndarrays
        indices = np.nonzero(x)[0]
        values = x[indices]

        if fill_to_max:
            num_base = self.bases.shape[1]
            random_basis_to_append = dict()
            i = 0
            while i + num_non_zero_entry < self.max_non_zero_entry:
                r = random.randint(0, num_base-1)
                # print(r, indices)
                if (r not in indices) and (r not in random_basis_to_append):
                    random_basis_to_append[r] = True
                    i += 1
            indices = np.r_[indices, np.array([key for key in random_basis_to_append.keys()])]
            values = np.r_[values, np.zeros(self.max_non_zero_entry - num_non_zero_entry)]
        return indices, values


def fit_wrapper(compressor_x):
    compressor, x = compressor_x
    return compressor.fit(x)


if __name__ == "__main__":

    all_emb = np.load("../data/dense_emb.npy")
    print(all_emb.shape)

    num_base = 1000
    base_emb = all_emb[:num_base]

    slv = Compressor(bases=base_emb.T, max_non_zero_entry=10)

    # For vectors in bases, alpha values are typically ~0.18
    for i in range(10):
        target_vector = all_emb[i, :]
        indices, values = slv.fit(target_vector, verbose=True)
        # indices is a 1-D np.ndarray
        # values is a 1-D np.ndarray
        print("non-zero entries:", indices, values)
        # print("non_zero entry: ", np.count_nonzero(x))
        # print(x)


    # Test multiprocessing module
    t1 = time.time()
    test_size = 1000
    debug = False
    # For those that are not in bases, alpha values are roughly 15 ~ 35
    for i in range(test_size):
        target_vector = all_emb[num_base+i, :]
        indices, values = slv.fit(target_vector, verbose=False)
        # indices is a 1-D np.ndarray
        # values is a 1-D np.ndarray
        if debug:
            print("non-zero entries:", indices, values)

    t2 = time.time()
    print("Sequential time: ", t2 - t1)

    def func(x):
        return slv.fit(x, verbose=False)

    # Direct parallel using Pool().map
    t1 = time.time()
    pool = multiprocessing.Pool(processes=2)
    ys = pool.map(func, all_emb[num_base:num_base+test_size, :])
    if debug:
        for ind, val in ys:
            print(ind, val)
    t2 = time.time()
    print("Parallel time: ", t2 - t1)

    # Parallel using a wrapper (because in model_without_mapfn.py, Compressor().fit() is not an out-most function
    # thus can't be pickled
    t1 = time.time()
    pool = multiprocessing.Pool(processes=2)
    parallel_params = [(slv, all_emb[i, :]) for i in range(num_base, num_base + test_size)]
    ys = pool.map(fit_wrapper, parallel_params)
    if debug:
        for ind, val in ys:
            print(ind, val)
    t2 = time.time()
    print("Parallel time: ", t2 - t1)
