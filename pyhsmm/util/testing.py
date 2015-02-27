from __future__ import division
import numpy as np
from numpy import newaxis as na
from matplotlib import pyplot as plt

import stats, general

#########################
#  statistical testing  #
#########################

### graphical

def populations_eq_quantile_plot(pop1, pop2, fig=None, percentilecutoff=5):
    pop1, pop2 = stats.flattendata(pop1), stats.flattendata(pop2)
    assert pop1.ndim == pop2.ndim == 1 or \
            (pop1.ndim == pop2.ndim == 2 and pop1.shape[1] == pop2.shape[1]), \
            'populations must have consistent dimensions'
    D = pop1.shape[1] if pop1.ndim == 2 else 1

    # we want to have the same number of samples
    n1, n2 = pop1.shape[0], pop2.shape[0]
    if n1 != n2:
        # subsample, since interpolation is dangerous
        if n1 < n2:
            pop1, pop2 = pop2, pop1
        np.random.shuffle(pop1)
        pop1 = pop1[:pop2.shape[0]]

    def plot_1d_scaled_quantiles(p1,p2,plot_midline=True):
        # scaled quantiles so that multiple calls line up
        p1.sort(), p2.sort() # NOTE: destructive! but that's cool
        xmin,xmax = general.scoreatpercentile(p1,percentilecutoff), \
                    general.scoreatpercentile(p1,100-percentilecutoff)
        ymin,ymax = general.scoreatpercentile(p2,percentilecutoff), \
                    general.scoreatpercentile(p2,100-percentilecutoff)
        plt.plot((p1-xmin)/(xmax-xmin),(p2-ymin)/(ymax-ymin))

        if plot_midline:
            plt.plot((0,1),(0,1),'k--')
        plt.axis((0,1,0,1))

    if D == 1:
        if fig is None:
            plt.figure()
        plot_1d_scaled_quantiles(pop1,pop2)
    else:
        if fig is None:
            fig = plt.figure()

        if not hasattr(fig,'_quantile_test_projs'):
            firsttime = True
            randprojs = np.random.randn(D,D)
            randprojs /= np.sqrt(np.sum(randprojs**2,axis=1))[:,na]
            projs = np.vstack((np.eye(D),randprojs))
            fig._quantile_test_projs = projs
        else:
            firsttime = False
            projs = fig._quantile_test_projs

        ims1, ims2 = pop1.dot(projs.T), pop2.dot(projs.T)
        for i, (im1, im2) in enumerate(zip(ims1.T,ims2.T)):
            plt.subplot(2,D,i)
            plot_1d_scaled_quantiles(im1,im2,plot_midline=firsttime)

### numerical

# NOTE: a random numerical test should be repeated at the OUTERMOST loop (with
# exception catching) to see if its failures exceed the number expected
# according to the specified pvalue (tests could be repeated via sample
# bootstrapping inside the test, but that doesn't work reliably and random tests
# should have no problem generating new randomness!)

def assert_populations_eq(pop1, pop2):
    assert_populations_eq_moments(pop1,pop2) and \
    assert_populations_eq_komolgorofsmirnov(pop1,pop2)

def assert_populations_eq_moments(pop1, pop2, **kwargs):
    # just first two moments implemented; others are hard to estimate anyway!
    assert_populations_eq_means(pop1,pop2,**kwargs) and \
    assert_populations_eq_variances(pop1,pop2,**kwargs)

def assert_populations_eq_means(pop1, pop2, pval=0.05, msg=None):
    _,p = stats.two_sample_t_statistic(pop1,pop2)
    if np.any(p < pval):
        raise AssertionError(msg or "population means might be different at %0.3f" % pval)

def assert_populations_eq_variances(pop1, pop2, pval=0.05, msg=None):
    _,p = stats.f_statistic(pop1, pop2)
    if np.any(p < pval):
        raise AssertionError(msg or "population variances might be different at %0.3f" % pval)

def assert_populations_eq_komolgorofsmirnov(pop1, pop2, msg=None):
    raise NotImplementedError # TODO

