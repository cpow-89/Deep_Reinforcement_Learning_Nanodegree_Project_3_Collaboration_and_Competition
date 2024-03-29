import numpy as np


class OnlineNormalizer(object):

    def __init__(self, nb_inputs):
        self.n = np.zeros((1, nb_inputs))
        self.mean = np.zeros((1, nb_inputs))
        self.mean_diff = np.zeros((1, nb_inputs))
        self.var = np.zeros((1, nb_inputs))

    def _observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        self._observe(inputs)
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
