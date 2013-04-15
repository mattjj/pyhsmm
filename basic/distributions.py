from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.special as special

from pybasicbayes.distributions import *
from abstractions import DurationDistribution


class PoissonDuration(Poisson, DurationDistribution):
    def __repr__(self):
        return 'PoissonDuration(lmbda=%0.2f,mean=%0.2f)' % (self.lmbda,self.lmbda+1)

    def log_sf(self,x):
        return stats.poisson.logsf(x-1,self.lmbda)

    def log_likelihood(self,x):
        return super(PoissonDuration,self).log_likelihood(x-1)

    def rvs(self,size=None):
        return super(PoissonDuration,self).rvs(size=size) + 1

    def _get_statistics(self,data):
        n, tot = super(PoissonDuration,self)._get_statistics(data)
        tot -= n
        return n, tot


class GeometricDuration(Geometric, DurationDistribution):
    def __repr__(self):
        return 'GeometricDuration(p=%0.2f)' % self.p

    def log_sf(self,x):
        return stats.geom.logsf(x,self.p)


class NegativeBinomialDuration(NegativeBinomial, DurationDistribution):
    def __repr__(self):
        return 'NegativeBinomialDuration(r=%0.2f,p=%0.2f)' % (self.r,self.p)

    def log_sf(self,x):
        return np.log(special.betainc(x,self.r,self.p))

    def log_likelihood(self,x):
        return super(NegativeBinomialDuration,self).log_likelihood(x-1)

    def rvs(self,size=None):
        return super(NegativeBinomialDuration,self).rvs(size=size) + 1

    def resample(self,data=[],*args,**kwargs):
        if isinstance(data,np.ndarray):
            return super(NegativeBinomialDuration,self).resample(data-1,*args,**kwargs)
        else:
            return super(NegativeBinomialDuration,self).resample([d-1 for d in data],*args,**kwargs)


# TODO need to get variants too... probably rewrite to use surgery

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

