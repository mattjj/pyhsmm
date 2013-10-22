from __future__ import division
import numpy as np
from IPython.parallel import Client

from util.general import engine_global_namespace

# NOTE: the ipcluster should be set up before this file is imported

### setup

c = Client(profile='lsf')
dv = c[:]
lbv = c.load_balanced_view()

dv.push(dict(my_data={}))

### util

def get_num_engines():
    return len(dv)

def phash(d):
    'hash based on object address in memory, not data values'
    assert isinstance(d,np.ndarray) or isinstance(d,tuple)
    if isinstance(d,np.ndarray):
        return d.__hash__()
    else:
        return hash(tuple(map(phash,d)))

def vhash(d):
    'hash based on data values'
    assert isinstance(d,np.ndarray) or isinstance(d,tuple)
    if isinstance(d,np.ndarray):
        d.flags.writeable = False
        return hash(d.data)
    else:
        return hash(tuple(map(vhash,d)))

### adding and managing data

# internals

# NOTE: data_id (and everything else that doesn't have to do with preloading) is
# based on phash, which should only be called on the controller

data_residency = {}
costs = np.zeros(len(dv))

@engine_global_namespace
def update_my_data(data_id,data):
    my_data[data_id] = data

# interface

def has_data(data):
    return phash(data) in data_residency

def add_data(data,costfunc=len):
    # NOTE: this is basically a one-by-one scatter with an additive parametric
    # cost function treated greedily
    ph = phash(data)
    engine_to_send = np.argmin(costs)
    data_residency[ph] = engine_to_send
    costs[engine_to_send] += costfunc(data)
    return c[engine_to_send].apply_async(update_my_data,ph,data)

def broadcast_data(data,costfunc=len):
    ph = phash(data)
    # sets data residency so that other functions can be used (one engine,
    # chosen by greedy static balancing, has responsibility)
    # NOTE: not blocking above assumes linear cost function
    engine_to_send = np.argmin(costs)
    data_residency[ph] = engine_to_send
    costs[engine_to_send] += costfunc(data)
    return dv.apply_async(update_my_data,ph,data)


def register_added_data(data):
    raise NotImplementedError # TODO

def register_broadcasted_data(data):
    raise NotImplementedError # TODO


def map_on_each(fn,added_datas,kwargss=None,engine_globals=None):
    @engine_global_namespace
    def _call(f,data_id,**kwargs):
        return f(my_data[data_id],**kwargs)

    if engine_globals is not None:
        dv.push(engine_globals,block=True)

    if kwargss is None:
        kwargss = [{} for data in added_datas] # no communication overhead

    indata = [(phash(data),data,kwargs) for data,kwargs in zip(added_datas,kwargss)]
    ars = [c[data_residency[data_id]].apply_async(_call,fn,data_id,**kwargs)
                    for data_id, data, kwargs in indata]
    dv.wait(ars)
    results = [ar.get() for ar in ars]

    c.purge_results('all')
    c.results.clear()
    dv.results.clear()

    return results

def map_on_each_broadcasted(fn,broadcasted_datas,kwargss=None,engine_globals=None):
    raise NotImplementedError # TODO lbv version

def call_with_all(fn,broadcasted_datas,kwargss,engine_globals=None):
    # one call for each element of kwargss
    @engine_global_namespace
    def _call(f,data_ids,kwargs):
        return f([my_data[data_id] for data_id in data_ids],**kwargs)

    if engine_globals is not None:
        dv.push(engine_globals,block=True)

    results = lbv.map_sync(
            _call,
            [fn]*len(kwargss),
            [[phash(data) for data in broadcasted_datas]]*len(kwargss),
            kwargss)

    c.purge_results('all')
    c.results.clear()
    dv.results.clear()

    return results

