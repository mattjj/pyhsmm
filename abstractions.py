import abc
import numpy as np
from warnings import warn

from pyhsmm.util.stats import combinedata
from matplotlib import pyplot as plt

class ObservationBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_likelihood(self,x):
        pass

    @abc.abstractmethod
    def resample(self,data=[]):
        # data is a (possibly masked) np.ndarray or list of (possibly masked)
        # np.ndarrays
        pass

    @abc.abstractmethod
    def rvs(self,size=[]):
        pass

    ### optional but recommended

    def plot(self,data=None,color='b'):
        raise NotImplementedError

    def test(self,*args,**kwargs):
        raise NotImplementedError


class DurationBase(object):
    '''
    Durations are like observations but with more restrictions and more requirements:
        - duration distributions can only be supported on positive integers
        - log_pmf is like log_likelihood, but an implementation of the
          log_survivor function is also required, which is defined by log_sf(x)
          = log(1-cdf(x))
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_pmf(self,x):
        pass

    def pmf(self,x):
        warn('Using generic implementation of %s.pmf().' % type(self))
        return np.exp(self.log_pmf(x))

    @abc.abstractmethod
    def log_sf(self,x):
        '''
        log survival function, defined by log_sf(x) = log(1-cdf(x)) where
        cdf(x) = P[X \leq x]

        in principle, an inefficient generic implementation could be based on
        log_pmf, but that seems too bad to ever encourage or allow
        '''
        pass

    @abc.abstractmethod
    def resample(self,data=[]):
        # data is a (possibly masked) np.ndarray or list of (possibly masked)
        # np.ndarrays
        pass

    @abc.abstractmethod
    def rvs(self,size=[]):
        pass

    ### optional but recommended

    def test(self,*args,**kwargs):
        raise NotImplementedError

    ### generic

    def plot(self,data=None,tmax=None,color='b'):
        if tmax is None:
            if data is not None:
                tmax = 2*data.max()
            else:
                tmax = 2*self.rvs(size=1000).mean() # TODO improve to log_sf less than something
        t = np.arange(1,tmax)
        plt.plot(t,self.pmf(t),color=color)

        if data is not None:
            plt.hist(data,bins=t-0.5,color=color,normed=True) # TODO only works with data as single array


class Collapsed(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def marginal_likelihood(self,data):
        # data is a (possibly masked) np.ndarray or list of (possibly masked)
        # np.ndarrays
        pass

    ### generic

    def predictive(self,newdata,olddata):
        # data is a (possibly masked) np.ndarray or list of (possibly masked)
        # np.ndarrays
        return np.exp(np.log(self.marginal_likelihood(combinedata((newdata,olddata))))
                - np.log(self.marginal_likelihood(olddata)))

