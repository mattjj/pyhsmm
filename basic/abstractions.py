from __future__ import division
import abc
import numpy as np
from matplotlib import pyplot as plt

from pybasicbayes.abstractions import *
from ..util.stats import flattendata

class DurationDistribution(Distribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_sf(self,x):
        '''
        log survival function, defined by log_sf(x) = log(1-cdf(x)) where
        cdf(x) = P[X \leq x]
        '''
        pass

    def pmf(self,x):
        return np.exp(self.log_pmf(x))

    def log_pmf(self,x):
        return self.log_likelihood(x)

    def plot(self,data=None,color='b'):
        data = flattendata(data) if data is not None else None

        try:
            tmax = np.where(np.exp(self.log_sf(np.arange(1,1000))) < 1e-3)[0][0]
        except IndexError:
            tmax = self.rvs(1000).mean()
        tmax = max(tmax,data.max()) if data is not None else tmax

        t = np.arange(1,tmax+1)
        plt.plot(t,self.pmf(t),color=color)

        if data is not None:
            if len(data) > 1:
                plt.hist(data,bins=t-0.5,color=color,normed=len(set(data)) > 1)
            else:
                plt.hist(data,bins=t-0.5,color=color)

