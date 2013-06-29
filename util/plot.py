from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

# TODO move pca to stats

def plot_gaussian_2D(mu, lmbda, color='b', centermarker=True,label=''):
    '''
    Plots mean and cov ellipsoid into current axes. Must be 2D. lmbda is a covariance matrix.
    '''
    assert len(mu) == 2

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)

    if centermarker:
        plt.plot([mu[0]],[mu[1]],marker='D',color=color,markersize=4)
    plt.plot(ellipse[0,:] + mu[0], ellipse[1,:] + mu[1],linestyle='-',linewidth=2,color=color,label=label)


def plot_gaussian_projection(mu, lmbda, vecs, **kwargs):
    '''
    Plots a ndim gaussian projected onto 2D vecs, where vecs is a matrix whose two columns
    are the subset of some orthonomral basis (e.g. from PCA on samples).
    '''
    plot_gaussian_2D(project_data(mu,vecs),project_ellipsoid(lmbda,vecs),**kwargs)


def pca_project_data(data,num_components=2):
    # convenience combination of the next two functions
    return project_data(data,pca(data,num_components=num_components))


def pca(data,num_components=2):
    U,s,Vh = np.linalg.svd(data - np.mean(data,axis=0))
    return Vh.T[:,:num_components]


def project_data(data,vecs):
    return np.dot(data,vecs.T)


def project_ellipsoid(ellipsoid,vecs):
    # vecs is a matrix whose columns are a subset of an orthonormal basis
    # ellipsoid is a pos def matrix
    return np.dot(vecs,np.dot(ellipsoid,vecs.T))


def subplot_gridsize(num):
    return sorted(min([(x,int(np.ceil(num/x))) for x in range(1,int(np.floor(np.sqrt(num)))+1)],key=sum))
