from __future__ import division
from builtins import zip, range
import numpy as np
from numpy.random import random
na = np.newaxis
import scipy.stats as stats
import scipy.special as special
import scipy.linalg
from numpy.core.umath_tests import inner1d

from . import general

# TODO write cholesky versions

### data abstraction

def atleast_2d(data):
    # NOTE: can't use np.atleast_2d because if it's 1D we want axis 1 to be the
    # singleton and axis 0 to be the sequence index
    if data.ndim == 1:
        return data.reshape((-1,1))
    return data

def mask_data(data):
    return np.ma.masked_array(np.nan_to_num(data),np.isnan(data),fill_value=0.,hard_mask=True)

def gi(data):
    return ~np.isnan(np.atleast_2d(data).reshape(data.shape[0],-1)).any(1)

def getdatasize(data):
    if isinstance(data,np.ma.masked_array):
        return data.shape[0] - data.mask.reshape((data.shape[0],-1))[:,0].sum()
    elif isinstance(data,np.ndarray):
        if len(data) == 0:
            return 0
        return data[gi(data)].shape[0]
    elif isinstance(data,list):
        return sum(getdatasize(d) for d in data)
    else:
        # handle unboxed case for convenience
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
        # handle unboxed case for convenience
        assert isinstance(data,int) or isinstance(data,float)
        return 1

def combinedata(datas):
    ret = []
    for data in datas:
        if isinstance(data,np.ma.masked_array):
            ret.append(np.ma.compress_rows(data))
        if isinstance(data,np.ndarray):
            ret.append(data)
        elif isinstance(data,list):
            ret.extend(combinedata(data))
        else:
            # handle unboxed case for convenience
            assert isinstance(data,int) or isinstance(data,float)
            ret.append(np.atleast_1d(data))
    return ret

def flattendata(data):
    # data is either an array (possibly a maskedarray) or a list of arrays
    if isinstance(data,np.ndarray):
        return data
    elif isinstance(data,list) or isinstance(data,tuple):
        if any(isinstance(d,np.ma.MaskedArray) for d in data):
            return np.concatenate([np.ma.compress_rows(d) for d in data])
        else:
            return np.concatenate(data)
    else:
        # handle unboxed case for convenience
        assert isinstance(data,int) or isinstance(data,float)
        return np.atleast_1d(data)

### misc

def getdata(l):
    return np.concatenate([x[gi(x)] for x in general.flatiter(l)])

def mean(datalist):
    return getdata(datalist).mean(0)

def cov(datalist):
    return np.cov(getdata(datalist),rowvar=0,bias=1)

def whiten(datalist):
    mu, L = mean(datalist), np.linalg.cholesky(cov(datalist))
    def apply_whitening(x):
        return np.linalg.solve(L, (x-mu).T).T + mu
    return general.treemap(apply_whitening, datalist)

def diag_whiten(datalist):
    mu, l = mean(datalist), np.sqrt(np.diag(cov(datalist)))
    def apply_whitening(x):
        return (x-mu)/l + mu
    return general.treemap(apply_whitening, datalist)

def count_transitions(stateseq, num_states):
    out = np.zeros((num_states,num_states),dtype=np.int32)
    for i,j in zip(stateseq[:-1],stateseq[1:]):
        out[i,j] += 1
    return out

### Sampling functions

def sample_discrete(distn,size=[],dtype=np.int32):
    'samples from a one-dimensional finite pmf'
    distn = np.atleast_1d(distn)
    assert (distn >=0).all() and distn.ndim == 1
    if (0 == distn).all():
        return np.random.randint(distn.shape[0],size=size)
    cumvals = np.cumsum(distn)
    return np.sum(np.array(random(size))[...,na] * cumvals[-1] > cumvals, axis=-1,dtype=dtype)

def sample_discrete_from_log(p_log,axis=0,dtype=np.int32):
    'samples log probability array along specified axis'
    cumvals = np.exp(p_log - np.expand_dims(p_log.max(axis),axis)).cumsum(axis) # cumlogaddexp
    thesize = np.array(p_log.shape)
    thesize[axis] = 1
    randvals = random(size=thesize) * \
            np.reshape(cumvals[[slice(None) if i is not axis else -1
                for i in range(p_log.ndim)]],thesize)
    return np.sum(randvals > cumvals,axis=axis,dtype=dtype)

def sample_markov(T,trans_matrix,init_state_distn):
    out = np.empty(T,dtype=np.int32)
    out[0] = sample_discrete(init_state_distn)
    for t in range(1,T):
        out[t] = sample_discrete(trans_matrix[out[t-1]])
    return out

def sample_niw(mu,lmbda,kappa,nu):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.
    '''
    # code is based on Matlab's method
    # reference: p. 87 in Gelman's Bayesian Data Analysis
    assert nu > lmbda.shape[0] and kappa > 0

    # first sample Sigma ~ IW(lmbda,nu)
    lmbda = sample_invwishart(lmbda,nu)
    # then sample mu | Lambda ~ N(mu, Lambda/kappa)
    mu = np.random.multivariate_normal(mu,lmbda / kappa)

    return mu, lmbda

def sample_invwishart(S,nu):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    # TODO lowmem! memoize! dchud (eigen?)
    n = S.shape[0]
    chol = np.linalg.cholesky(S)

    if (nu <= 81+n) and (nu == np.round(nu)):
        x = np.random.randn(nu,n)
    else:
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(nu-np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return np.dot(T,T.T)

def sample_wishart(sigma, nu):
    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (nu <= 81+n) and (nu == round(nu)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,nu)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(nu - np.arange(n))))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)

def sample_mn(M,U=None,Uinv=None,V=None,Vinv=None):
    assert (U is None) ^ (Uinv is None)
    assert (V is None) ^ (Vinv is None)

    G = np.random.normal(size=M.shape)

    if U is not None:
        G = np.dot(np.linalg.cholesky(U),G)
    else:
        G = np.linalg.solve(np.linalg.cholesky(Uinv).T,G)

    if V is not None:
        G = np.dot(G,np.linalg.cholesky(V).T)
    else:
        G = np.linalg.solve(np.linalg.cholesky(Vinv).T,G.T).T

    return M + G

def sample_mniw(nu,S,M,K=None,Kinv=None):
    assert (K is None) ^ (Kinv is None)
    Sigma = sample_invwishart(S,nu)
    if K is not None:
        return sample_mn(M=M,U=Sigma,V=K), Sigma
    else:
        return sample_mn(M=M,U=Sigma,Vinv=Kinv), Sigma

def sample_pareto(x_m,alpha):
    return x_m + np.random.pareto(alpha)

def sample_crp_tablecounts(concentration,customers,colweights):
    m = np.zeros_like(customers)
    tot = customers.sum()
    randseq = np.random.random(tot)

    starts = np.empty_like(customers)
    starts[0,0] = 0
    starts.flat[1:] = np.cumsum(np.ravel(customers)[:customers.size-1])

    for (i,j), n in np.ndenumerate(customers):
        w = colweights[j]
        for k in range(n):
            m[i,j] += randseq[starts[i,j]+k] \
                    < (concentration * w) / (k + concentration * w)

    return m

### Entropy
def invwishart_entropy(sigma,nu,chol=None):
    D = sigma.shape[0]
    chol = np.linalg.cholesky(sigma) if chol is None else chol
    Elogdetlmbda = special.digamma((nu-np.arange(D))/2).sum() + D*np.log(2) - 2*np.log(chol.diagonal()).sum()
    return invwishart_log_partitionfunction(sigma,nu,chol)-(nu-D-1)/2*Elogdetlmbda + nu*D/2

def invwishart_log_partitionfunction(sigma,nu,chol=None):
    D = sigma.shape[0]
    chol = np.linalg.cholesky(sigma) if chol is None else chol
    return -1*(nu*np.log(chol.diagonal()).sum() - (nu*D/2*np.log(2) + D*(D-1)/4*np.log(np.pi) \
            + special.gammaln((nu-np.arange(D))/2).sum()))

### Predictive

def multivariate_t_loglik(y,nu,mu,lmbda):
    # returns the log value
    d = len(mu)
    yc = np.array(y-mu,ndmin=2)
    L = np.linalg.cholesky(lmbda)
    ys = scipy.linalg.solve_triangular(L,yc.T,overwrite_b=True,lower=True)
    return scipy.special.gammaln((nu+d)/2.) - scipy.special.gammaln(nu/2.) \
            - (d/2.)*np.log(nu*np.pi) - np.log(L.diagonal()).sum() \
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

