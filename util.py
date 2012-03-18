from __future__ import division
import numpy as np
import cPickle, itertools, collections

def one_vs_all(stuff):
    stuffset = set(stuff)
    for thing in stuff:
        yield thing, stuffset - set([thing])

def rle(stateseq):
    stateseq = np.array(stateseq)
    pos, = np.where(np.diff(stateseq) != 0)
    pos = np.concatenate(([0],pos+1,[len(stateseq)]))
    return stateseq[pos[:-1]], np.diff(pos)

def medianfilt(times,vals,timewindow,subsample=1):
    # TODO instead of subsample, maybe this should be expressed in terms of
    # samples per unit time!
    '''
    output[t] is the median of all vals within centered timewindow of t
    '''
    assert times.ndim == 1
    assert times.shape == vals.shape
    out = np.zeros(len(times)//subsample)
    outtimes = np.zeros(out.shape)
    # TODO could make faster by making a max lookahead, possibly by checking the
    # max index diff in times within the windowsize
    for idx in xrange(len(times)//subsample):
        outtimes[idx] = t = times[idx*subsample]
        out[idx] = np.median(vals[(t - timewindow/2 <= times) & (times <= t + timewindow/2)])
    assert not np.isnan(out).any()
    return outtimes, out

def deepcopy(obj):
    return cPickle.loads(cPickle.dumps(obj))

notifier = None
def notifydone(message='Python finished a process'):
    global notifier
    if notifier is None:
        import Growl
        notifier = Growl.GrowlNotifier(notifications=['Dynamic Messages'],applicationIcon=Growl.Image.imageWithIconForApplication('Terminal'))
        notifier.register()
    notifier.notify('Dynamic Messages','Work Complete!',message)

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
