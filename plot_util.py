import numpy as np
from matplotlib import pyplot as plt
from stats_util import project_ellipsoid

def plot_gaussian_2D(mu, lmbda, color='b', centermarker=True):
    '''
    Plots mean and cov ellipsoid into current axes. Must be 2D. lmbda is a covariance matrix.
    '''
    assert len(mu) == 2

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)

    if centermarker:
        plt.plot([mu[0]],[mu[1]],color + 'D',markersize=4)
    plt.plot(ellipse[0,:] + mu[0], ellipse[1,:] + mu[1],color+'-')


def plot_gaussian_projection(mu, lmbda, vecs, **kwargs):
    '''
    Plots a ndim gaussian projected onto 2D vecs, where vecs is a matrix whose two columns 
    are the subset of some orthonomral basis (e.g. from PCA on samples).
    '''
    plot_gaussian_2D(np.dot(vecs.T,mu),project_ellipsoid(lmbda,vecs),**kwargs)


