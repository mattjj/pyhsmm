from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.special as special

from pybasicbayes.distributions import *
from pybasicbayes.models import MixtureDistribution
from abstractions import DurationDistribution

# If you're not comfortable with metaprogramming, here be dragons

##########################
#  Metaprogramming util  #
##########################

def _make_duration_distribution(cls):
    class Wrapper(cls, DurationDistribution):
        pass

    Wrapper.__name__ = cls.__name__ + 'Duration'
    Wrapper.__doc__ = cls.__doc__
    return Wrapper

# this method is for generating new classes, shifting their support from
# {0,1,2,...} to {1,2,3,...}
def _start_at_one(cls):
    class Wrapper(cls, DurationDistribution):
        def log_likelihood(self,x,*args,**kwargs):
            return super(Wrapper,self).log_likelihood(x-1,*args,**kwargs)

        def log_sf(self,x,*args,**kwargs):
            return super(Wrapper,self).log_sf(x-1,*args,**kwargs)

        def rvs(self,size=None):
            return super(Wrapper,self).rvs(size)+1

        def resample(self,data=[],*args,**kwargs):
            if isinstance(data,np.ndarray):
                return super(Wrapper,self).resample(data-1,*args,**kwargs)
            else:
                return super(Wrapper,self).resample([d-1 for d in data],*args,**kwargs)

        def max_likelihood(self,data,weights=None,*args,**kwargs):
            if weights is not None:
                raise NotImplementedError
            else:
                if isinstance(data,np.ndarray):
                    return super(Wrapper,self).max_likelihood(data-1,weights=None,*args,**kwargs)
                else:
                    return super(Wrapper,self).max_likelihood([d-1 for d in data],weights=None,*args,**kwargs)

    Wrapper.__name__ = cls.__name__ + 'Duration'
    Wrapper.__doc__ = cls.__doc__
    return Wrapper

##########################
#  Distribution classes  #
##########################

class GeometricDuration(Geometric, DurationDistribution):
    pass # mixin style!

PoissonDuration = _start_at_one(Poisson)

NegativeBinomialDuration = _start_at_one(NegativeBinomial)
NegativeBinomialFixedRDuration = _start_at_one(NegativeBinomialFixedR)
NegativeBinomialIntegerRDuration = _start_at_one(NegativeBinomialIntegerR)
NegativeBinomialVariantDuration = _make_duration_distribution(NegativeBinomialVariant)
NegativeBinomialFixedRVariantDuration = _make_duration_distribution(NegativeBinomialFixedRVariant)
NegativeBinomialIntegerRVariantDuration = _make_duration_distribution(NegativeBinomialIntegerRVariant)

#################
#  Model stuff  #
#################

# this is extending the MixtureDistribution from basic/pybasicbayes/models.py
# and then clobbering the name
class MitureDistribution(MixtureDistribution, DurationDistribution):
    # TODO test this
    def log_sf(self,x):
        x = np.asarray(x,dtype=np.float64)
        K = len(self.components)
        vals = np.empty((x.shape[0],K))
        for idx, c in enumerate(self.components):
            vals[:,idx] = c.log_sf(x)
        vals += self.weights.log_likelihood(np.arange(K))
        return np.logaddexp.reduce(vals,axis=1)

##########
#  Meta  #
##########

# this class is for delaying instances of duration distributions
class Delay(DurationDistribution):
    def __init__(self,dur_distn,delay):
        self.dur_distn = dur_distn
        self.delay = delay

    def log_sf(self,x):
        return self.dur_distn.log_sf(x-self.delay)

    def log_likelihood(self,x):
        return self.dur_distn.log_likelihood(x-self.delay)

    def rvs(self,size=None):
        return self.dur_distn.rvs(size=size) + self.delay

    def resample(self,data=[],*args,**kwargs):
        if isinstance(data,np.ndarray):
            return self.dur_distn.resample(data-self.delay,*args,**kwargs)
        else:
            return self.dur_distn.resample([d-self.delay for d in data],*args,**kwargs)

    def max_likelihood(self,*args,**kwargs):
        raise NotImplementedError

