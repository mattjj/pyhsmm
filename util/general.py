from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import scipy.linalg
import copy, collections, os, shutil
from contextlib import closing
from urllib2 import urlopen
from itertools import izip, chain, count, ifilter

def solve_psd(A,b,chol=None,overwrite_b=False,overwrite_A=False):
    if A.shape[0] < 5000 and chol is None:
        return np.linalg.solve(A,b)
    else:
        if chol is None:
            chol = np.linalg.cholesky(A)
        return scipy.linalg.solve_triangular(
                chol.T,
                scipy.linalg.solve_triangular(chol,b,lower=True,overwrite_b=overwrite_b),
                lower=False,overwrite_b=True)

def interleave(*iterables):
    return list(chain.from_iterable(zip(*iterables)))

def joindicts(dicts):
    # stuff on right clobbers stuff on left
    return reduce(lambda x,y: dict(x,**y), dicts, {})

def one_vs_all(stuff):
    stuffset = set(stuff)
    for thing in stuff:
        yield thing, stuffset - set([thing])

def rle(stateseq):
    pos, = np.where(np.diff(stateseq) != 0)
    pos = np.concatenate(([0],pos+1,[len(stateseq)]))
    return stateseq[pos[:-1]], np.diff(pos)

def irle(vals,lens):
    out = np.empty(np.sum(lens))
    for v,l,start in zip(vals,lens,np.concatenate(((0,),np.cumsum(lens)[:-1]))):
        out[start:start+l] = v
    return out

def ibincount(counts):
    'returns an array a such that counts = np.bincount(a)'
    return np.repeat(np.arange(counts.shape[0]),counts)

def cumsum(v,strict=False):
    if not strict:
        return np.cumsum(v,axis=0)
    else:
        out = np.zeros_like(v)
        out[1:] = np.cumsum(v[:-1],axis=0)
        return out

def rcumsum(v,strict=False):
    if not strict:
        return np.cumsum(v[::-1],axis=0)[::-1]
    else:
        out = np.zeros_like(v)
        out[:-1] = np.cumsum(v[-1:0:-1],axis=0)[::-1]
        return out

def delta_like(v,i):
    out = np.zeros_like(v)
    out[i] = 1
    return out

def deepcopy(obj):
    return copy.deepcopy(obj)

def nice_indices(arr):
    '''
    takes an array like [1,1,5,5,5,999,1,1]
    and maps to something like [0,0,1,1,1,2,0,0]
    modifies original in place as well as returns a ref
    '''
    # surprisingly, this is slower for very small (and very large) inputs:
    # u,f,i = np.unique(arr,return_index=True,return_inverse=True)
    # arr[:] = np.arange(u.shape[0])[np.argsort(f)][i]
    ids = collections.defaultdict(count().next)
    for idx,x in enumerate(arr):
        arr[idx] = ids[x]
    return arr

def ndargmax(arr):
    return np.unravel_index(np.argmax(np.ravel(arr)),arr.shape)

def match_by_overlap(a,b):
    assert a.ndim == b.ndim == 1 and a.shape[0] == b.shape[0]
    ais, bjs = list(set(a)), list(set(b))
    scores = np.zeros((len(ais),len(bjs)))
    for i,ai in enumerate(ais):
        for j,bj in enumerate(bjs):
            scores[i,j] = np.dot(np.array(a==ai,dtype=np.float),b==bj)

    flip = len(bjs) > len(ais)

    if flip:
        ais, bjs = bjs, ais
        scores = scores.T

    matching = []
    while scores.size > 0:
        i,j = ndargmax(scores)
        matching.append((ais[i],bjs[j]))
        scores = np.delete(np.delete(scores,i,0),j,1)
        ais = np.delete(ais,i)
        bjs = np.delete(bjs,j)

    return matching if not flip else [(x,y) for y,x in matching]

def hamming_error(a,b):
    return (a!=b).sum()

def scoreatpercentile(data,per,axis=0):
    'like the function in scipy.stats but with an axis argument and works on arrays'
    a = np.sort(data,axis=axis)
    idx = per/100. * (data.shape[axis]-1)

    if (idx % 1 == 0):
        return a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]]
    else:
        lowerweight = 1-(idx % 1)
        upperweight = (idx % 1)
        idx = int(np.floor(idx))
        return lowerweight * a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]] \
                + upperweight * a[[slice(None) if ii != axis else idx+1 for ii in range(a.ndim)]]

def stateseq_hamming_error(sampledstates,truestates):
    sampledstates = np.array(sampledstates,ndmin=2).copy()

    errors = np.zeros(sampledstates.shape[0])
    for idx,s in enumerate(sampledstates):
        # match labels by maximum overlap
        matching = match_by_overlap(s,truestates)
        s2 = s.copy()
        for i,j in matching:
            s2[s==i] = j
        errors[idx] = hamming_error(s2,truestates)

    return errors if errors.shape[0] > 1 else errors[0]

def _sieve(stream):
    # just for fun; doesn't work over a few hundred
    val = stream.next()
    yield val
    for x in ifilter(lambda x: x%val != 0, _sieve(stream)):
        yield x

def primes():
    return _sieve(count(2))

def top_eigenvector(A,niter=1000,force_iteration=False):
    '''
    assuming the LEFT invariant subspace of A corresponding to the LEFT
    eigenvalue of largest modulus has geometric multiplicity of 1 (trivial
    Jordan block), returns the vector at the intersection of that eigenspace and
    the simplex

    A should probably be a ROW-stochastic matrix

    probably uses power iteration
    '''
    n = A.shape[0]
    np.seterr(invalid='raise',divide='raise')
    if n <= 25 and not force_iteration:
        x = np.repeat(1./n,n)
        x = np.linalg.matrix_power(A.T,niter).dot(x)
        x /= x.sum()
        return x
    else:
        x1 = np.repeat(1./n,n)
        x2 = x1.copy()
        for itr in xrange(niter):
            np.dot(A.T,x1,out=x2)
            x2 /= x2.sum()
            x1,x2 = x2,x1
            if np.linalg.norm(x1-x2) < 1e-8:
                break
        return x1

def engine_global_namespace(f):
    # see IPython.parallel.util.interactive; it's copied here so as to avoid
    # extra imports/dependences elsewhere, and to provide a slightly clearer
    # name
    f.__module__ = '__main__'
    return f

def block_view(a,block_shape):
    shape = (a.shape[0]/block_shape[0],a.shape[1]/block_shape[1]) + block_shape
    strides = (a.strides[0]*block_shape[0],a.strides[1]*block_shape[1]) + a.strides
    return ast(a,shape=shape,strides=strides)

def count_transitions(stateseq,minlength=None):
    if minlength is None:
        minlength = stateseq.max() + 1
    out = np.zeros((minlength,minlength),dtype=np.int32)
    for a,b in izip(stateseq[:-1],stateseq[1:]):
        out[a,b] += 1
    return out

### SGD

def sgd_steps(tau,kappa):
    assert 0.5 < kappa <= 1 and tau >= 0
    for t in count(1):
        yield (t+tau)**(-kappa)

def hold_out(datalist,frac):
    N = len(datalist)
    perm = np.random.permutation(N)
    split = int(np.ceil(frac * N))
    return [datalist[i] for i in perm[split:]], [datalist[i] for i in perm[:split]]

def sgd_passes(tau,kappa,datalist,minibatchsize=1,npasses=1):
    N = len(datalist)

    for superitr in xrange(npasses):
        if minibatchsize == 1:
            perm = np.random.permutation(N)
            for idx, rho_t in izip(perm,sgd_steps(tau,kappa)):
                yield datalist[idx], rho_t
        else:
            minibatch_indices = np.array_split(np.random.permutation(N),N/minibatchsize)
            for indices, rho_t in izip(minibatch_indices,sgd_steps(tau,kappa)):
                yield [datalist[idx] for idx in indices], rho_t

def sgd_sampling(tau,kappa,datalist,minibatchsize=1):
    N = len(datalist)
    if minibatchsize == 1:
        for rho_t in sgd_steps(tau,kappa):
            minibatch_index = np.random.choice(N)
            yield datalist[minibatch_index], rho_t
    else:
        for rho_t in sgd_steps(tau,kappa):
            minibatch_indices = np.random.choice(N,size=minibatchsize,replace=False)
            yield [datalist[idx] for idx in minibatch_indices], rho_t

# TODO should probably eliminate this function
def minibatchsize(lst):
    return float(sum(d.shape[0] for d in lst))

### misc

def random_subset(lst,sz):
    perm = np.random.permutation(len(lst))
    return [lst[perm[idx]] for idx in xrange(sz)]

def get_file(remote_url,local_path):
    if not os.path.isfile(local_path):
        with closing(urlopen(remote_url)) as remotefile:
            with open(local_path,'wb') as localfile:
                shutil.copyfileobj(remotefile,localfile)

def list_split(lst,num):
    assert num > 0
    return [lst[start::num] for start in range(num)]

def indicators_to_changepoints(indseq,which='ends'):
    shift = 1 if which == 'ends' else 0
    changes = list(shift + np.where(indseq)[0])

    if changes[0] != 0:
        changes.insert(0,0)
    if changes[-1] != len(indseq):
        changes.append(len(indseq))

    return zip(changes[:-1],changes[1:])

def indices_to_changepoints(T,changes):
    changes = list(changes)

    if changes[0] != 0:
        changes.insert(0,0)
    if changes[-1] != T:
        changes.append(T)

    return zip(changes[:-1],changes[1:])

