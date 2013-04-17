from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.special as special

from pybasicbayes.distributions import *
from abstractions import DurationDistribution

# If you're not comfortable with metaprogramming, here be dragons

##########################
#  Metaprogramming util  #
##########################

# this method is for generating new classes, shifting their support from
# {0,1,2,...} to {1,2,3,...}
def _start_at_one(cls):
    class Wrapper(cls, DurationDistribution):
        def log_likelihood(self,x):
            return super(Wrapper,self).log_likelihood(x-1)

        def log_sf(self,x):
            return super(Wrapper,self).log_sf(x-1)

        def rvs(self,size=None):
            return super(Wrapper,self).rvs(size)+1

        def resample(self,data=[],*args,**kwargs):
            if isinstance(data,np.ndarray):
                return super(Wrapper,self).resample(data-1,*args,**kwargs)
            else:
                return super(Wrapper,self).resample([d-1 for d in data],*args,**kwargs)

        def max_likelihood(self,*args,**kwargs):
            raise NotImplementedError

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
NegativeBinomialVariantDuration = _start_at_one(NegativeBinomialVariant)
NegativeBinomialFixedRVariantDuration = _start_at_one(NegativeBinomialFixedRVariant)
NegativeBinomialIntegerRVariantDuration = _start_at_one(NegativeBinomialIntegerRVariant)

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

