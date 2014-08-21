from __future__ import division
import abc
import numpy as np
from matplotlib import pyplot as plt

from pybasicbayes.abstractions import *
from ..util.stats import flattendata, sample_discrete_from_log, combinedata
from ..util.general import rcumsum

class DurationDistribution(Distribution):
    __metaclass__ = abc.ABCMeta

    # in addition to the methods required by Distribution, we also require a
    # log_sf implementation

    @abc.abstractmethod
    def log_sf(self,x):
        '''
        log survival function, defined by log_sf(x) = log(P[X \gt x]) =
        log(1-cdf(x)) where cdf(x) = P[X \leq x]
        '''
        pass

    def log_pmf(self,x):
        return self.log_likelihood(x)

    def expected_log_pmf(self,x):
        return self.expected_log_likelihood(x)

    # default implementations below

    def pmf(self,x):
        return np.exp(self.log_pmf(x))

    def rvs_given_greater_than(self,x):
        tail = self.log_sf(x)
        if np.isinf(tail):
            return x+1
        trunc = 500
        while self.log_sf(x+trunc) - tail > -20:
            trunc = int(1.1*trunc)
        logprobs = self.log_pmf(np.arange(x+1,x+trunc+1)) - tail
        return sample_discrete_from_log(logprobs)+x+1

    def expected_log_sf(self,x):
        x = np.atleast_1d(x).astype('int32')
        assert x.ndim == 1
        inf = max(2*x.max(),2*1000) # approximately infinity, we hope
        return rcumsum(self.expected_log_pmf(np.arange(1,inf)),strict=True)[x]

    def resample_with_truncations(self,data=[],truncated_data=[]):
        '''
        truncated_data is full of observations that were truncated, so this
        method samples them out to be at least that large
        '''
        if not isinstance(truncated_data,list):
            filled_in = np.asarray([self.rvs_given_greater_than(x-1) for x in truncated_data])
        else:
            filled_in = np.asarray([self.rvs_given_greater_than(x-1)
                for xx in truncated_data for x in xx])
        self.resample(data=combinedata((data,filled_in)))

    @property
    def mean(self):
        trunc = 500
        while self.log_sf(trunc) > -20:
            trunc *= 1.5
        return np.arange(1,trunc+1).dot(self.pmf(np.arange(1,trunc+1)))

    def plot(self,data=None,color='b',**kwargs):
        data = flattendata(data) if data is not None else None

        try:
            tmax = np.where(np.exp(self.log_sf(np.arange(1,1000))) < 1e-3)[0][0]
        except IndexError:
            tmax = 2*self.rvs(1000).mean()
        tmax = max(tmax,data.max()) if data is not None else tmax

        t = np.arange(1,tmax+1)
        plt.plot(t,self.pmf(t),color=color)

        if data is not None:
            if len(data) > 1:
                plt.hist(data,bins=t-0.5,color=color,normed=len(set(data)) > 1)
            else:
                plt.hist(data,bins=t-0.5,color=color)

