import abc
import numpy as np
from warnings import warn

class ObservationBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_likelihood(self,x):
        pass

    @abc.abstractmethod
    def resample(self,data=np.array([])):
        # data is a (possibly masked) np.ndarray or list of (possibly masked)
        # np.ndarrays
        pass

    @abc.abstractmethod
    def rvs(self,size=[]):
        pass

    def plot(self,data=None,color='b'):
        raise NotImplementedError

    def test(self,**kwargs):
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
    def resample(self,data=np.array([])):
        # data is a (possibly masked) np.ndarray or list of (possibly masked)
        # np.ndarrays
        pass

    @abc.abstractmethod
    def rvs(self,size=[]):
        pass


class Collapsed(object):
    __metaclass__ = abc.ABCMeta

    def predictive(self,newdata,olddata=np.array([])):
        # data is a (possibly masked) np.ndarray or list of (possibly masked)
        # np.ndarrays
        olddata.shape = (-1,) + newdata.shape[1:] if newdata.ndim > 1 else olddata.shape
        return np.exp(np.log(self.marginal_likelihood(np.concatenate((newdata,olddata))))
                - np.log(self.marginal_likelihood(olddata)))

    @abc.abstractmethod
    def marginal_likelihood(self,data):
        # data is a (possibly masked) np.ndarray or list of (possibly masked)
        # np.ndarrays
        pass

