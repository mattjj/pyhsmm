from __future__ import division
import numpy as np
from numpy import newaxis as na
import scipy.stats as stats
from matplotlib import pyplot as plt
from warnings import warn

from pyhsmm.abstractions import ObservationBase
from pyhsmm.util.stats import sample_niw, sample_discrete, sample_discrete_from_log


'''
This module includes general distribution classes that can be used in sampling
hierarchical Bayesian models. Though module is called 'observations', there
aren't really any restrictions placed on an observation distribution, so these
classes can be reused as general distributions.
'''

class mixture(ObservationBase):
    '''
    This class is for mixtures of other observation distributions,
    reusing the multinomial class.
    member data are:
        1. alpha: prior parameters for w. if it is a number, equivalent to
           passing alpha*np.ones(num_components)
        2. components: a list of distribution objects representing the mixture
        3. weights: a vector specifying the weight of each mixture component
    '''
    def __repr__(self):
        n_mix = self.n_mix
        display = 'number of mixture: %s\n' % n_mix
        display += 'log weight and parameters for mixture k:\n'
        for k in range(n_mix):
           display += 'log weight: %s ' % self.weights.log_likelihood(k)
           display += self.components[k].__repr__() + '\n'
        return display

    def __init__(self,alpha,components,weights=None):
        self.n_mix = n_mix = len(components)

        alpha = np.array(alpha)
        assert alpha.ndim == 0 or alpha.ndim == 1
        if alpha.ndim == 0:
            alpha = alpha * np.ones(n_mix)
        else:
            assert len(alpha) == n_mix

        self.components = components
        self.weights = multinomial(alpha)

        if weights is not None:
            self.weights.discrete = weights

    def _log_scores(self,x):
        '''score for component i on data j is in retval[i,j]'''
        return self.weights.log_likelihood(np.arange(self.n_mix))[:,na] + \
                np.vstack([c.log_likelihood(x) for c in self.components])

    def resample(self,data=np.array([]),niter=1,**kwargs):
        n = float(len(data))
        if n == 0:
            self.weights.resample()
            for c in self.components:
                c.resample()
        else:
            for itr in range(niter):
                # sample labels
                log_scores = self._log_scores(data)
                labels = sample_discrete_from_log(log_scores,axis=0)

                # resample weights
                self.weights.resample(labels)

                # resample component parameters
                for idx, c in enumerate(self.components):
                    c.resample(data[labels == idx])

    def log_likelihood(self,x):
        return np.logaddexp.reduce(self._log_scores(x),axis=0)

    def rvs(self,size=[]):
        size = np.array(size,ndmin=1)
        labels = self.weights.rvs(size=int(np.prod(size)))
        counts = np.bincount(labels)
        out = np.concatenate([c.rvs(size=(count,)) for count,c in zip(counts,self.components)],axis=0)
        out = out[np.random.permutation(len(out))] # maybe this shuffle isn't formally necessary
        return np.reshape(out,np.concatenate((size,(-1,))))

    @classmethod
    def test(cls):
        foo = cls(alpha=3.,components=[gaussian(np.zeros(2),np.eye(2),0.02,4) for idx in range(4)])
        data = foo.rvs(200)

        bar = cls(alpha=2./8,components=[gaussian(np.zeros(2),np.eye(2),0.02,4) for idx in range(8)])
        bar.resample(data,niter=50)

        plt.plot(data[:,0],data[:,1],'kx')
        for c,weight in zip(bar.components,bar.weights.discrete):
            if weight > 0.1:
                plt.plot(c.mu[0],c.mu[1],'bo',markersize=10)
        plt.show()

    def plot(self):
        raise NotImplementedError

# convenience method, TODO move elsewhere
def gaussian_mixture(alpha_vec=np.array([3, 3]),
        mu_0=np.zeros(39),lmbda_0=np.eye(39),kappa_0=5.,nu_0=49.):
    return mixture(alpha=alpha_vec,components=[gaussian(mu_0,lmbda_0,kappa_0,nu_0)])


def diagonal_gaussian_mixture(alpha_vec=np.array([3, 3]),
        mu_0=np.zeros(39),nus_0=np.ones(39)*0.05,alphas_0=10*np.ones(39),betas_0=np.ones(39)):
    return mixture(alpha=alpha_vec,components=[diagonal_gaussian(mu_0,nus_0, alphas_0, betas_0) for x in range(len(alpha_vec))])

class gaussian(ObservationBase):
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

    def __init__(self,mu_0,lmbda_0,kappa_0,nu_0,mu=None,sigma=None):
        self.nu_0 = nu_0
        self.kappa_0 = kappa_0
        self.mu_0 = mu_0
        self.lmbda_0 = lmbda_0

        if mu is None or sigma is None:
            self.resample()
        else:
            self.mu = mu
            self.sigma = sigma

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

    @classmethod
    def _plot_setup(cls,instance_list):
        # must set cls.vecs to be a reasonable 2D space to project onto
        # so that the projection is consistent across instances
        # for now, i'll just make it random if there are more than 2 dimensions
        assert len(instance_list) > 0
        assert len(set([len(o.mu) for o in instance_list])) == 1, 'must have consistent dimensions across instances'
        dim = len(instance_list[0].mu)
        if dim > 2:
            vecs = np.random.randn((dim,2))
            vecs /= np.sqrt((vecs**2).sum())
        else:
            vecs = np.eye(2)

        for o in instance_list:
            o.global_vecs = vecs

    def plot(self,data=None,color='b'):
        from pyhsmm.util.plot import project_data, plot_gaussian_projection, pca
        # if global projection vecs exist, use those
        # otherwise, when dim>2, do a pca on the data
        try:
            vecs = self.global_vecs
        except AttributeError:
            dim = len(self.mu)
            if dim == 2:
                vecs = np.eye(2)
            else:
                assert dim > 2
                vecs = pca(data,num_components=2)

        if data is not None:
            projected_data = project_data(data,vecs)
            plt.plot(projected_data[:,0],projected_data[:,1],marker='.',linestyle=' ',color=color)

        plot_gaussian_projection(self.mu,self.sigma,vecs,color=color)


# TODO can I make nonconjugate versions of next two classes by setting nu
# parameter to zero?

class diagonal_gaussian(gaussian):
    '''
    product of normal-inverse-gamma priors over mu (mean vector) and sigmas
    (vector of scalar variances).

    the prior follows
        sigmas     ~ InvGamma(alpha_0,beta_0) iid
        mu | sigma ~ N(mu_0,1/nu_0 * diag(sigmas))
    '''
    def __init__(self,mu_0,nus_0,alphas_0,betas_0,mu=None,sigmas=None):
        self.mu_0 = mu_0
        # all the s's refer to the fact that these are vectors of length
        # len(mu_0)
        self.nus_0 = nus_0
        self.alphas_0 = alphas_0
        self.betas_0 = betas_0

        if mu is None or sigmas is None:
            self.resample()
        else:
            self.mu = mu
            self.sigmas = sigmas

    def resample(self,data=np.array([]),**kwargs):
        n = float(len(data))
        k = len(self.mu_0)
        if n == 0:
            self.sigmas = stats.invgamma.rvs(self.alphas_0,scale=self.betas_0)
            self.mu = np.sqrt(self.sigmas/self.nus_0)*np.random.randn(k)+self.mu_0
        else:
            xbar = data.mean(0)
            nus_n = n + self.nus_0
            alphas_n = self.alphas_0 + n/2
            betas_n = self.betas_0 + 1./2 * ((data-xbar)**2).sum(0) + n*self.nus_0/(n+self.nus_0)\
                    * 1./2 * (data.mean(0) - self.mu_0)**2
            mu_n = (n*data.mean(0) + self.nus_0*self.mu_0) / (n + self.nus_0)

            self.sigmas = stats.invgamma.rvs(alphas_n,scale=betas_n)
            self.mu = np.sqrt(self.sigmas / nus_n) * np.random.randn(k)+mu_n

    def rvs(self,size=[]):
        size = np.array(size,ndmin=1)
        return np.sqrt(self.sigmas)*np.random.normal(size=np.concatenate((size,self.mu.shape))) + self.mu

    def log_likelihood(self,x,mu=None,sigmas=None):
        if sigmas is None:
            sigmas = self.sigmas
        if mu is None:
            mu = self.mu
        return (-0.5*((x-self.mu)**2/self.sigmas) - np.log(np.sqrt(2*np.pi*self.sigmas))).sum(1)


class isotropic_gaussian(gaussian):
    '''
    normal-inverse-gamma prior over mu (mean vector) and sigma (scalar
    variance). essentially, all coordinates of all observations inform the
    variance.

    the prior follows
        sigma      ~ InvGamma(alpha_0,beta_0)
        mu | sigma ~ N(mu_0,sigma/nu_0 * I)
    '''
    def __init__(self,mu_0,nu_0,alpha_0,beta_0,mu=None,sigma=None):
        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if mu is None or sigma is None:
            self.resample()
        else:
            self.mu = mu
            self.sigma = sigma

    def resample(self,data=np.array([]),**kwargs):
        n = float(len(data))
        k = len(self.mu_0)
        if n == 0:
            self.sigma = stats.invgamma.rvs(self.alpha_0,scale=self.beta_0)
            self.mu = np.sqrt(self.sigma/self.nu_0)*np.random.randn(k)+self.mu_0
        else:
            xbar = data.mean(0)
            nu_n = k*n + self.nu_0
            alpha_n = self.alpha_0 + k*n/2
            beta_n = self.beta_0 + 1./2 * np.linalg.norm(data - xbar,'fro')**2 + \
                    (n*k*self.nu_0)/(n*k+self.nu_0) * \
                    1./2 * ((xbar - self.mu_0)**2).sum() # derive by factoring in each column at a time
            mu_n = (n*xbar + self.nu_0*self.mu_0) / (n + self.nu_0)

            self.sigma = stats.invgamma.rvs(alpha_n,scale=beta_n)
            self.mu = np.sqrt(self.sigma/nu_n)*np.random.randn(k)+mu_n

    def rvs(self,size=[]):
        size = np.array(size,ndmin=1)
        return np.sqrt(self.sigma)*np.random.normal(size=np.concatenate((size,self.mu.shape))) + self.mu

    def log_likelihood(self,x,mu=None,sigma=None):
        if sigma is None:
            sigma = self.sigma
        if mu is None:
            mu = self.mu
        k = len(mu)
        return (-0.5*((x-self.mu)**2).sum(1)/self.sigma - k*np.log(np.sqrt(2*np.pi*self.sigma)))


class multinomial(ObservationBase):
    '''
    This class represents a multinomial distribution in a label form.
    For example, if len(alpha_vec) == 3, then five samples of data may look like
    [0,1,0,2,1]
    Each entry is the label of a sample, like the outcome of die rolls.

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

class scalar_gaussian_nonconj(ObservationBase):
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

class scalar_gaussian_nonconj_gelparams(ObservationBase):
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

