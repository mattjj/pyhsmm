from __future__ import division
import numpy as np
from numpy.random import random
na = np.newaxis
import scipy.stats as stats
import scipy.special as special
import scipy.linalg
from numpy.core.umath_tests import inner1d

import general

# TODO write cholesky versions

### data abstraction

def getdatasize(data):
    if isinstance(data,np.ndarray):
        return data.shape[0]
    elif isinstance(data,list):
        return sum(getdatasize(d) for d in data)
    else:
        assert isinstance(data,int) or isinstance(data,float)
        return 1

def getdatadimension(data):
    if isinstance(data,np.ndarray):
        assert data.ndim > 1
        return data.shape[1]
    elif isinstance(data,list):
        assert len(data) > 0
        return getdatadimension(data[0])
    else:
        assert isinstance(data,int) or isinstance(data,float)
        return 1

def combinedata(datas):
    ret = []
    for data in datas:
        if isinstance(data,np.ndarray):
            ret.append(data)
        elif isinstance(data,list):
            ret.extend(data)
        else:
            assert isinstance(data,int) or isinstance(data,float)
            ret.append(np.atleast_1d(data))
    return ret

def flattendata(data):
    # data is either an array (possibly a maskedarray) or a list of arrays
    if isinstance(data,np.ndarray):
        return data
    elif isinstance(data,list) or isinstance(data,tuple):
        if any(isinstance(d,np.ma.MaskedArray) for d in data):
            return np.ma.concatenate(data).compressed()
        else:
            return np.concatenate(data)
    else:
        # handle unboxed case for convenience
        assert isinstance(data,int) or isinstance(data,float)
        return np.atleast_1d(data)

### misc

def cov(a):
    # return np.cov(a,rowvar=0,bias=1)
    mu = a.mean(0)
    return a.T.dot(a)/a.shape[0] - np.outer(mu,mu)

### Sampling functions

def sample_discrete(distn,size=[],dtype=np.int):
    'samples from a one-dimensional finite pmf'
    distn = np.atleast_1d(distn)
    assert (distn >=0).all() and distn.ndim == 1
    cumvals = np.cumsum(distn)
    return np.sum(random(size)[...,na] * cumvals[-1] > cumvals, axis=-1,dtype=dtype)

def sample_discrete_from_log(p_log,axis=0,dtype=np.int):
    'samples log probability array along specified axis'
    cumvals = np.exp(p_log - np.expand_dims(p_log.max(axis),axis)).cumsum(axis) # cumlogaddexp
    thesize = np.array(p_log.shape)
    thesize[axis] = 1
    randvals = random(size=thesize) * \
            np.reshape(cumvals[[slice(None) if i is not axis else -1
                for i in range(p_log.ndim)]],thesize)
    return np.sum(randvals > cumvals,axis=axis,dtype=dtype)

def sample_niw(mu,lmbda,kappa,nu):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.
    '''
    # code is based on Matlab's method
    # reference: p. 87 in Gelman's Bayesian Data Analysis

    # first sample Sigma ~ IW(lmbda,nu)
    lmbda = sample_invwishart(lmbda,nu)
    # then sample mu | Lambda ~ N(mu, Lambda/kappa)
    mu = np.random.multivariate_normal(mu,lmbda / kappa)

    return mu, lmbda

def sample_invwishart(lmbda,dof):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    # TODO lowmem! memoize! dchud (eigen?)
    n = lmbda.shape[0]
    chol = np.linalg.cholesky(lmbda)

    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(dof,n)
    else:
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(dof-np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return np.dot(T,T.T)

def sample_wishart(sigma, dof):
    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (dof <= 81+n) and (dof == round(dof)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(n))))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)

def sample_mn(Sigma,M,K):
    left = np.linalg.cholesky(Sigma)
    right = np.linalg.cholesky(K)
    return M + left.dot(np.random.normal(size=M.shape)).dot(right.T)

def sample_mniw(dof,lmbda,M,K):
    Sigma = sample_invwishart(lmbda,dof)
    return sample_mn(Sigma,M,K), Sigma

### Entropy
def invwishart_entropy(sigma,nu,chol=None):
    D = sigma.shape[0]
    chol = general.cholesky(sigma) if chol is None else chol
    Elogdetlmbda = special.digamma((nu-np.arange(D))/2).sum() + D*np.log(2) - 2*np.log(chol.diagonal()).sum()
    return invwishart_log_partitionfunction(sigma,nu,chol)-(nu-D-1)/2*Elogdetlmbda + nu*D/2

def invwishart_log_partitionfunction(sigma,nu,chol=None):
    D = sigma.shape[0]
    chol = general.cholesky(sigma) if chol is None else chol
    return -1*(nu*np.log(chol.diagonal()).sum() - (nu*D/2*np.log(2) + D*(D-1)/4*np.log(np.pi) \
            + special.gammaln((nu-np.arange(D))/2).sum()))

### Predictive

def multivariate_t_loglik(y,nu,mu,lmbda):
    # returns the log value
    d = len(mu)
    yc = np.array(y-mu,ndmin=2)
    ys, LT = general.solve_chofactor_system(lmbda,yc.T,overwrite_b=True)
    return scipy.special.gammaln((nu+d)/2.) - scipy.special.gammaln(nu/2.) \
            - (d/2.)*np.log(nu*np.pi) - np.log(LT.diagonal()).sum() \
            - (nu+d)/2.*np.log1p(1./nu*inner1d(ys.T,ys.T))

def beta_predictive(priorcounts,newcounts):
    prior_nsuc, prior_nfail = priorcounts
    nsuc, nfail = newcounts

    numer = scipy.special.gammaln(np.array([nsuc+prior_nsuc,
        nfail+prior_nfail, prior_nsuc+prior_nfail])).sum()
    denom = scipy.special.gammaln(np.array([prior_nsuc, prior_nfail,
        prior_nsuc+prior_nfail+nsuc+nfail])).sum()
    return numer - denom

### Statistical tests

def two_sample_t_statistic(pop1, pop2):
    pop1, pop2 = (flattendata(p) for p in (pop1, pop2))
    t = (pop1.mean(0) - pop2.mean(0)) / np.sqrt(pop1.var(0)/pop1.shape[0] + pop2.var(0)/pop2.shape[0])
    p = 2*stats.t.sf(np.abs(t),np.minimum(pop1.shape[0],pop2.shape[0]))
    return t,p

def f_statistic(pop1, pop2): # TODO test
    pop1, pop2 = (flattendata(p) for p in (pop1, pop2))
    var1, var2 = pop1.var(0), pop2.var(0)
    n1, n2 = np.where(var1 >= var2, pop1.shape[0], pop2.shape[0]), \
             np.where(var1 >= var2, pop2.shape[0], pop1.shape[0])
    var1, var2 = np.maximum(var1,var2), np.minimum(var1,var2)
    f = var1 / var2
    p = stats.f.sf(f,n1,n2)
    return f,p

