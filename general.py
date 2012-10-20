from __future__ import division
import numpy as np
import cPickle, itertools, collections

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

def deepcopy(obj):
    # there's a library function that does the same thing, consider replacing
    # this function with that one
    return cPickle.loads(cPickle.dumps(obj))

def nice_indices(arr):
    '''
    takes an array like [1,1,5,5,5,999,1,1]
    and maps to something like [0,0,1,1,1,2,0,0]
    modifies original in place as well as returns a ref
    '''
    ids = collections.defaultdict(itertools.count().next)
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
            scores[i,j] = np.dot(a==ai,b==bj)

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

def scoreatpercentile(data,per,axis):
    '''
    like the function in scipy.stats but with an axis argument, and works on
    arrays.
    '''
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
    val = stream.next()
    yield val
    for x in itertools.ifilter(lambda x: x%val != 0, _sieve(stream)):
        yield x

def primes():
    return _sieve(itertools.count(2))
