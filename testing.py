from __future__ import division

import stats

#########################
#  statistical testing  #
#########################

### graphical

def populations_eq_plot(pop1, pop2):
    # TODO plot sorted against each other
    # if multidimensional, do each coordinate as well as some random projections
    raise NotImplementedError

### numerical

# NOTE: a random numerical test should be repeated at the OUTERMOST loop to see
# if its failures exceed the number expected according to the specified pvalue
# (tests could be repeated via sample bootstrapping inside the test, but that
# doesn't work reliably and random tests should have no problem generating new
# randomness!)

def assert_populations_eq(pop1, pop2):
    assert_populations_eq_moments(pop1,pop2) and \
    assert_populations_eq_komolgorofsmirnov(pop1,pop2)

def assert_populations_eq_moments(pop1, pop2, **kwargs):
    # just first two moments implemented; others are hard to estimate anyway!
    assert_populations_eq_means(pop1,pop2,**kwargs) and \
    assert_populations_eq_variances(pop1,pop2,**kwargs)

def assert_populations_eq_means(pop1, pop2, pval=0.05, msg=None):
    _,p = stats.two_sample_t_statistic(pop1,pop2)
    if p < pval:
        raise AssertionError(msg or "population means might be different at %0.3f" % pval)

def assert_populations_eq_variances(pop1, pop2, pval=0.05, msg=None):
    _,p = stats.f_statistic(pop1, pop2)
    if p < pval:
        raise AssertionError(msg or "population variances might be different at %0.3f" % pval)

def assert_populations_eq_komolgorofsmirnov(pop1, pop2, msg=None):
    raise NotImplementedError # TODO

