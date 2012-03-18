from __future__ import division
import numpy as np
from numpy import newaxis as na

import scipy.stats as stats
from stats_util import sample_niw, sample_discrete

from matplotlib import pyplot as plt

class gaussian(object):
    '''
    Multivariate Gaussian observation distribution class. NOTE: Only
    works for 2 or more dimensions. For a scalar Gaussian, use one of the scalar
    classes.
    Uses a conjugate Normal/Inverse-Wishart prior.

    Hyperparameters follow Gelman et al.'s notation in Bayesian Data
    Analysis:
    nu_0, lmbda_0
    mu_0, kappa_0

    Parameters are mean and covariance matrix:
    mu, sigma
    '''

    def __repr__(self):
        return 'gaussian(mu=%s,sigma=%s,nu_0=%s,lmbda_0=%s,mu_0=%s,kappa_0=%s)' % (self.mu,self.sigma,self.nu_0,self.lmbda_0,self.mu_0,self.kappa_0)

    def __init__(self,mu=None, sigma=None, nu_0=10, kappa_0=0.05, mu_0=np.zeros(10), lmbda_0=np.eye(10)):
        self.nu_0 = nu_0
        self.kappa_0 = kappa_0
        self.mu_0 = mu_0
        self.lmbda_0 = lmbda_0

        if mu is None or sigma is None:
            self.resample()
        else:
            self.mu = mu
            self.sigma = sigma
            # center at given inputs
            self.mu_0 = mu
            self.lmbda_0 = sigma
            self.nu_0 = sigma.shape[0]

    def resample(self,data=np.array([]),**kwargs):
        n = float(len(data))
        if n == 0:
            self.mu, self.sigma = sample_niw(self.mu_0, self.lmbda_0, self.kappa_0, self.nu_0)
        else:
            # calculate sufficient statistics
            xbar = np.mean(data,axis=0)
            centered = data - xbar
            sumsq = np.dot(centered.T,centered)
            # form posterior hyperparameters
            mu_n = self.kappa_0 / (self.kappa_0 + n) * self.mu_0 + n / (self.kappa_0 + n) * xbar
            kappa_n = self.kappa_0 + n
            nu_n = self.nu_0 + n
            lmbda_n = self.lmbda_0 + sumsq + self.kappa_0*n/(self.kappa_0+n) * np.outer(xbar-self.mu_0,xbar-self.mu_0)
            # sample with those hyperparameters
            self.mu, self.sigma = sample_niw(mu_n, lmbda_n, kappa_n, nu_n)

    def log_likelihood(self,x,mu=None,sigma=None):
        if mu is None or sigma is None:
            mu, sigma = self.mu, self.sigma
        obs_dim = float(len(mu))
        x = np.array(x,ndmin=2) - mu
        return -1./2. * np.sum(x * np.linalg.solve(sigma,x.T).T,axis=1) - np.log((2*np.pi)**(obs_dim / 2.) * np.sqrt(np.linalg.det(sigma)))

    def rvs(self,size=[]):
        return np.random.multivariate_normal(mean=self.mu,cov=self.sigma,size=size)


class multinomial(object):
    '''
    This class represents a multinomial distribution in a label form.
    For example, if len(alpha_vec) == 3, then five samples of data may look like
    [0,1,0,2,1]
    Each entry is the label of a sample, like the outcome of dice rolls.

    Hyperparameters: alpha_vec
    Parameters: discrete, which is a vector encoding of a discrete
    probability distribution
    '''

    def __init__(self,alpha_vec,discrete=None):
        self.alpha_vec = alpha_vec
        if discrete is not None:
            assert len(discrete) == len(alpha_vec)
            self.discrete = discrete
        else:
            self.resample()

    def resample(self,data=np.array([]),**kwargs):
        assert data.ndim == 1
        if data.size == 0:
            counts = np.zeros(self.alpha_vec.shape)
        else:
            # too bad bincount won't accept length-zero arrays!
            counts = np.bincount(data,minlength=len(self.alpha_vec))
        self._resample_given_counts(counts)
        
    def _resample_given_counts(self,counts):
        self.discrete = stats.gamma.rvs(self.alpha_vec + counts)
        self.discrete /= self.discrete.sum()
        assert not np.isnan(self.discrete).any()

    def log_likelihood(self,x):
        assert x.ndim == 1
        return np.log(self.discrete)[x]

    def rvs(self,size=[]):
        return sample_discrete(self.discrete,size)

    @classmethod
    def test(cls,num_tests=4):
        alpha_vec = 0.5*np.ones(6)
        t = np.arange(0,len(alpha_vec))
        fig = plt.figure()
        fig.suptitle('log_likelihood and rvs consistency')
        for idx in range(num_tests):
            plt.subplot(num_tests,1,idx+1)
            o = cls(alpha_vec)
            data = o.rvs(100)
            plt.hist(data,np.arange(-0.5,0.5+len(alpha_vec),1),normed=True)
            plt.plot(t,np.exp(o.log_likelihood(t)),'-',marker='.')
        
        fig = plt.figure()
        fig.suptitle('posterior sampling correctness')
        for idx in range(num_tests):
            ogen = cls(alpha_vec)
            data = ogen.rvs(100)

            oinfer = cls(alpha_vec)
            plt.subplot(num_tests,2,2*idx+1)
            plt.hist(data,np.arange(-0.5,0.5+len(alpha_vec),1),normed=True)
            plt.plot(t,np.exp(oinfer.log_likelihood(t)),'r-',marker='.')
            plt.title('before resampling')

            plt.subplot(num_tests,2,2*idx+2)
            oinfer.resample(data)
            plt.hist(data,np.arange(-0.5,0.5+len(alpha_vec),1),normed=True)
            plt.plot(t,np.exp(oinfer.log_likelihood(t)),'g-',marker='.')
            plt.title('after resampling')

class indicator_multinomial(multinomial):
    '''
    This class represents a multinomial distribution in an indicator/count form.
    For example, if len(alpha_vec) == 3, then five samples worth of indicator
    data may look like
    [[0,1,0],
     [1,0,0],
     [1,0,0],
     [0,0,1],
     [0,1,0]]
    Each row is an indicator of a sample, and summing over rows gives counts.

    Based on the way the methods are written, the data rows may also be count
    arrays themselves. The same sample set as in the previous example can also
    be represented as

    [[2,2,1]]

    or

    [[1,1,1],
     [1,1,0]]

    etc.

    Hyperparameters: alpha_vec
    Parameters: discrete, which is a vector encoding of a discrete
    probability distribution
    '''

    def resample(self,data=np.array([]),**kwargs):
        if data.size == 0:
            counts = np.zeros(self.alpha_vec.shape)
        elif data.ndim == 2:
            counts = data.sum(0)
        else:
            counts = data
        self._resample_given_counts(counts)

    def log_likelihood(self,x):
        assert x.ndim == 2
        assert x.shape[1] == len(self.discrete)
        return (x * np.log(self.discrete)).sum(1)

    def rvs(self,size=0):
        assert type(size) == type(0)
        label_data = multinomial.rvs(self,size=size)
        out = np.zeros((size,len(self.alpha_vec)))
        out[np.arange(out.shape[0]),label_data] = 1
        return out

    @classmethod
    def test(cls):
        # I've tested this by hand
        raise NotImplementedError

class scalar_gaussian_nonconj(object):
    def __init__(self,mu_0,sigmasq_0,alpha,beta,mu=None,sigmasq=None,mubin=None,sigmasqbin=None):
        self.mu_0 = mu_0
        self.sigmasq_0 = sigmasq_0
        self.alpha = alpha
        self.beta = beta

        self.mubin = mubin
        self.sigmasqbin = sigmasqbin

        if mu is None or sigmasq is None:
            self.resample()
        else:
            self.mu = mu
            self.sigmasq = sigmasq

    def resample(self,data=np.array([[]]),niter=10):
        if data.size == 0:
            # sample from prior
            self.mu = np.sqrt(self.sigmasq_0)*np.random.randn()+self.mu_0
            self.sigmasq = stats.invgamma.rvs(self.alpha,scale=self.beta)
        else:
            assert data.ndim == 2
            assert data.shape[1] == 1
            n = len(data)
            for iter in xrange(niter):
                # resample mean given data and var
                mu_n = (self.mu_0/self.sigmasq_0 + data.sum()/self.sigmasq)/(1/self.sigmasq_0 + n/self.sigmasq)
                sigmasq_n = 1/(1/self.sigmasq_0 + n/self.sigmasq)
                self.mu = np.sqrt(sigmasq_n)*np.random.randn()+mu_n
                #resample variance given data and mean
                alpha_n = self.alpha+n/2
                beta_n = self.beta+((data-self.mu)**2).sum()/2
                self.sigmasq = stats.invgamma.rvs(alpha_n,scale=beta_n)

        if self.mubin is not None and self.sigmasqbin is not None:
            self.mubin[...] = self.mu
            self.sigmasqbin[...] = self.sigmasq

    def rvs(self,size=None):
        return np.sqrt(self.sigmasq)*np.random.normal(size=size)+self.mu

    def log_likelihood(self,x):
        assert x.ndim == 2
        assert x.shape[1] == 1
        return (-0.5*(x-self.mu)**2/self.sigmasq - np.log(np.sqrt(2*np.pi*self.sigmasq))).flatten()

    def __repr__(self):
        return 'gaussian_scalar_nonconj(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)

class scalar_gaussian_nonconj_gelparams(object):
    # uses parameters from Gelman's Bayesian Data Analysis
    def __init__(self,mu_0,tausq_0,sigmasq_0,nu_0,mu=None,sigmasq=None,mubin=None,sigmasqbin=None):
        self.mu_0 = mu_0
        self.tausq_0 = tausq_0
        self.sigmasq_0 = sigmasq_0
        self.nu_0 = nu_0

        self.mubin = mubin
        self.sigmasqbin = sigmasqbin

        if mu is None or sigmasq is None:
            self.resample()
        else:
            self.mu = mu
            self.sigmasq = sigmasq
            if mubin is not None and sigmasqbin is not None:
                mubin[...] = mu
                sigmasqbin[...] = sigmasq

    def resample(self,data=np.array([[]]),niter=10):
        if data.size == 0:
            # sample from prior
            self.mu = np.sqrt(self.tausq_0)*np.random.randn()+self.mu_0
            self.sigmasq = self.nu_0 * self.sigmasq_0 / stats.chi2.rvs(self.nu_0)
        else:
            assert data.ndim == 2
            assert data.shape[1] == 1
            n = len(data)
            mu_hat = data.mean()
            for iter in xrange(niter):
                # resample mean given data and var
                mu_n = (self.mu_0/self.tausq_0 + n*mu_hat/self.sigmasq)/(1/self.tausq_0 + n/self.sigmasq)
                tausq_n = 1/(1/self.tausq_0 + n/self.sigmasq)
                self.mu = np.sqrt(tausq_n)*np.random.randn()+mu_n
                #resample variance given data and mean
                v = np.var(data - self.mu)
                nu_n = self.nu_0 + n
                sigmasq_n = (self.nu_0 * self.sigmasq_0 + n*v)/(self.nu_0 + n)
                self.sigmasq = nu_n * sigmasq_n / stats.chi2.rvs(nu_n)

        if self.mubin is not None and self.sigmasqbin is not None:
            self.mubin[...] = self.mu
            self.sigmasqbin[...] = self.sigmasq

    def rvs(self,size=None):
        return np.sqrt(self.sigmasq)*np.random.normal(size=size)+self.mu

    def log_likelihood(self,x):
        assert x.ndim == 2
        assert x.shape[1] == 1
        return (-0.5*(x-self.mu)**2/self.sigmasq - np.log(np.sqrt(2*np.pi*self.sigmasq))).flatten()

    def __repr__(self):
        return 'gaussian_scalar_nonconj(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)

class scalar_gaussian_nonconj_fixedvar(scalar_gaussian_nonconj_gelparams):
    def __init__(self,mu_0,tausq_0,sigmasq,mu=None,mubin=None,sigmasqbin=None):
        self.mu_0 = mu_0
        self.tausq_0 = tausq_0
        self.sigmasq = sigmasq

        self.mubin = mubin

        # only set once
        if sigmasqbin is not None:
            sigmasqbin[...] = sigmasq

        if mu is None:
            self.resample()
        else:
            self.mu = mu
            if mubin is not None:
                mubin[...] = mu

    def resample(self,data=np.array([[]])):
        if data.size == 0:
            self.mu = np.sqrt(self.tausq_0)*np.random.randn()+self.mu_0
        else:
            n = len(data)
            mu_hat = data.mean()
            mu_n = (self.mu_0/self.tausq_0 + n*mu_hat/self.sigmasq)/(1/self.tausq_0 + n/self.sigmasq)
            tausq_n = 1/(1/self.tausq_0 + n/self.sigmasq)
            self.mu = np.sqrt(tausq_n)*np.random.randn()+mu_n

        if self.mubin is not None:
            self.mubin[...] = self.mu


class scalar_gaussian_conj(scalar_gaussian_nonconj_gelparams):
    def __init__(self,mu_0,kappa_0,sigmasq_0,nu_0,mubin=None,sigmasqbin=None):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.sigmasq_0 = sigmasq_0
        self.nu_0 = nu_0

        self.mubin = mubin
        self.sigmasqbin = sigmasqbin

        self.resample()

    def resample(self,data=np.array([[]]),**kwargs):
        if data.size < 2:
            self.sigmasq = self.nu_0 * self.sigmasq_0 / stats.chi2.rvs(self.nu_0)
            self.mu = np.sqrt(self.sigmasq / self.kappa_0) * np.random.randn() + self.mu_0
        else:
            ybar = data.mean()
            n = len(data)
            sumsq = ((data-ybar)**2).sum()

            mu_n = (self.kappa_0 * self.mu_0 + n * ybar) / (self.kappa_0 + n)
            kappa_n = self.kappa_0 + n
            nu_n = self.nu_0 + n
            nu_n_sigmasq_n = self.nu_0 * self.sigmasq_0 + sumsq + self.kappa_0 * n / (self.kappa_0 + n) * (ybar - self.mu_0)**2

            self.sigmasq = nu_n_sigmasq_n / stats.chi2.rvs(nu_n)
            self.mu = np.sqrt(self.sigmasq / kappa_n) * np.random.randn() + mu_n

        if self.mubin is not None and self.sigmasqbin is not None:
            self.mubin[...] = self.mu
            self.sigmasqbin[...] = self.sigmasq

