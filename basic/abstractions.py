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

    def plot(self,data=None,tmax=None,color='b'):
        data = flattendata(data)
        if tmax is None:
            if data is not None:
                tmax = 2*data.max()
            else:
                tmax = 2*self.rvs(size=1000).mean() # TODO improve to log_sf less than something
        t = np.arange(1,tmax)
        plt.plot(t,self.pmf(t),color=color)

        if data is not None:
            plt.hist(data,bins=t-0.5,color=color,normed=True) # TODO only works with data as single array

