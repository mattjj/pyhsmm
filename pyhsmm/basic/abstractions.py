from __future__ import division
from future.utils import with_metaclass

import abc
import numpy as np
from matplotlib import pyplot as plt

from pybasicbayes.abstractions import *
import pyhsmm
from pyhsmm.util.stats import flattendata, sample_discrete, sample_discrete_from_log, combinedata
from pyhsmm.util.general import rcumsum

class DurationDistribution(with_metaclass(abc.ABCMeta, Distribution)):

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

        # if numerical underflow, return anything sensible
        if np.isinf(tail):
            return x+1

        # if big tail, rejection sample
        elif np.exp(tail) > 0.1:
            y = self.rvs(25)
            while not np.any(y > x):
                y = self.rvs(25)
            return y[y > x][0]

        # otherwise, sample directly using the pmf and sf
        else:
            u = np.random.rand()
            y = x
            while u > 0:
                u -= np.exp(self.log_pmf(y) - tail)
                y += 1
            return y

    def rvs_given_less_than(self,x,num):
        pmf = self.pmf(np.arange(1,x))
        return sample_discrete(pmf,num)+1

    def expected_log_sf(self,x):
        x = np.atleast_1d(x).astype('int32')
        assert x.ndim == 1
        inf = max(2*x.max(),2*1000) # approximately infinity, we hope
        return rcumsum(self.expected_log_pmf(np.arange(1,inf)),strict=True)[x]

    def resample_with_censoring(self,data=[],censored_data=[]):
        '''
        censored_data is full of observations that were censored, meaning a
        value of x really could have been anything >= x, so this method samples
        them out to be at least that large
        '''
        filled_in = self._uncensor_data(censored_data)
        return self.resample(data=combinedata((data,filled_in)))

    def _uncensor_data(self,censored_data):
        # TODO numpy-vectorize this!
        if len(censored_data) > 0:
            if not isinstance(censored_data,list):
                filled_in = np.asarray([self.rvs_given_greater_than(x-1)
                    for x in censored_data])
            else:
                filled_in = np.asarray([self.rvs_given_greater_than(x-1)
                    for xx in censored_data for x in xx])
        else:
            filled_in = []
        return filled_in

    def resample_with_censoring_and_truncation(self,data=[],censored_data=[],left_truncation_level=None):
        filled_in = self._uncensor_data(censored_data)

        if left_truncation_level is not None and left_truncation_level > 1:
            norm = self.pmf(np.arange(1,left_truncation_level)).sum()
            num_rejected = np.random.geometric(1-norm)-1
            rejected_observations = self.rvs_given_less_than(left_truncation_level,num_rejected) \
                    if num_rejected > 0 else []
        else:
            rejected_observations = []

        self.resample(data=combinedata((data,filled_in,rejected_observations)))

    @property
    def mean(self):
        # TODO this is dumb, why is this here?
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

