import numpy as np


class DualAveraging:
    def __init__(
            self,
            x0: float,
            mu: float,
            gamma: float,
            t0: int,
            omega: float,
            maxval: float=np.inf,
            minval: float=-np.inf
    ):
        self.x = x0
        self.xb = x0
        self.mu = mu
        self.gamma = gamma
        self.t0 = t0
        self.omega = omega
        self.t = 0
        self.summ = 0.0
        self.maxval, self.minval = maxval, minval

    @property
    def eta(self):
        return self.t**-self.omega

    def update(self, new_term: float):
        self.t += 1
        rt = np.sqrt(self.t) / (self.gamma * (self.t + self.t0))
        self.summ += new_term
        self.x = self.mu - rt*self.summ
        self.x = np.clip(self.x, self.minval, self.maxval)
        eta = self.eta
        self.xb = eta*self.x + (1-eta)*self.xb

class RuppertAveraging:
    def __init__(
            self,
            x0: float,
            omega: float,
            maxval: float=np.inf,
            minval: float=-np.inf
    ):
        self.x = x0
        self.xb = x0
        self.omega = omega
        self.t = 0
        self.maxval, self.minval = maxval, minval

    @property
    def eta(self):
        return self.t**-self.omega

    def update(self, new_term: float):
        self.t += 1
        self.x -= self.eta*new_term
        self.x = np.clip(self.x, self.minval, self.maxval)
        self.xb = self.t / (self.t + 1) * self.xb + self.x / (self.t + 1)
