from __future__ import division
import numpy as np
from scipy.special import logsumexp

from pybasicbayes.distributions import *
from pybasicbayes.models import MixtureDistribution
from .abstractions import DurationDistribution

##############################################
#  Mixins for making duratino distributions  #
##############################################

class _StartAtOneMixin(object):
    def log_likelihood(self,x,*args,**kwargs):
        return super(_StartAtOneMixin,self).log_likelihood(x-1,*args,**kwargs)

    def log_sf(self,x,*args,**kwargs):
        return super(_StartAtOneMixin,self).log_sf(x-1,*args,**kwargs)

    def expected_log_likelihood(self,x,*args,**kwargs):
        return super(_StartAtOneMixin,self).expected_log_likelihood(x-1,*args,**kwargs)

    def rvs(self,size=None):
        return super(_StartAtOneMixin,self).rvs(size)+1

    def rvs_given_greater_than(self,x):
        return super(_StartAtOneMixin,self).rvs_given_greater_than(x)+1

    def resample(self,data=[],*args,**kwargs):
        if isinstance(data,np.ndarray):
            return super(_StartAtOneMixin,self).resample(data-1,*args,**kwargs)
        else:
            return super(_StartAtOneMixin,self).resample([d-1 for d in data],*args,**kwargs)

    def max_likelihood(self,data,weights=None,*args,**kwargs):
        if isinstance(data,np.ndarray):
            return super(_StartAtOneMixin,self).max_likelihood(
                    data-1,weights=weights,*args,**kwargs)
        else:
            return super(_StartAtOneMixin,self).max_likelihood(
                    [d-1 for d in data],weights=weights,*args,**kwargs)

    def meanfieldupdate(self,data,weights,*args,**kwargs):
        if isinstance(data,np.ndarray):
            return super(_StartAtOneMixin,self).meanfieldupdate(
                    data-1,weights=weights,*args,**kwargs)
        else:
            return super(_StartAtOneMixin,self).meanfieldupdate(
                    [d-1 for d in data],weights=weights,*args,**kwargs)

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        if isinstance(data,np.ndarray):
            return super(_StartAtOneMixin,self).meanfield_sgdstep(
                    data-1,weights=weights,
                    prob=prob,stepsize=stepsize)
        else:
            return super(_StartAtOneMixin,self).meanfield_sgdstep(
                    [d-1 for d in data],weights=weights,
                    prob=prob,stepsize=stepsize)

##########################
#  Distribution classes  #
##########################

class GeometricDuration(
        Geometric,
        DurationDistribution):
    pass

class PoissonDuration(
        _StartAtOneMixin,
        Poisson,
        DurationDistribution):
    pass

class NegativeBinomialDuration(
        _StartAtOneMixin,
        NegativeBinomial,
        DurationDistribution):
    pass

class NegativeBinomialFixedRDuration(
        _StartAtOneMixin,
        NegativeBinomialFixedR,
        DurationDistribution):
    pass

class NegativeBinomialIntegerRDuration(
        _StartAtOneMixin,
        NegativeBinomialIntegerR,
        DurationDistribution):
    pass

class NegativeBinomialIntegerR2Duration(
        _StartAtOneMixin,
        NegativeBinomialIntegerR2,
        DurationDistribution):
    pass

class NegativeBinomialFixedRVariantDuration(
        NegativeBinomialFixedRVariant,
        DurationDistribution):
    pass

class NegativeBinomialIntegerRVariantDuration(
        NegativeBinomialIntegerRVariant,
        DurationDistribution):
    pass

#################
#  Model stuff  #
#################

# this is extending the MixtureDistribution from basic/pybasicbayes/models.py
# and then clobbering the name
class MixtureDistribution(MixtureDistribution, DurationDistribution):
    # TODO test this
    def log_sf(self,x):
        x = np.asarray(x,dtype=np.float64)
        K = len(self.components)
        vals = np.empty((x.shape[0],K))
        for idx, c in enumerate(self.components):
            vals[:,idx] = c.log_sf(x)
        vals += self.weights.log_likelihood(np.arange(K))
        return logsumexp(vals,axis=1)

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
        return self.dur_distn.rvs(size) + self.delay

    def resample(self,data=[],*args,**kwargs):
        if isinstance(data,np.ndarray):
            return self.dur_distn.resample(data-self.delay,*args,**kwargs)
        else:
            return self.dur_distn.resample([d-self.delay for d in data],*args,**kwargs)

    def max_likelihood(self,*args,**kwargs):
        raise NotImplementedError

