from __future__ import division
import numpy as np
from IPython.parallel import Client
import os
from util.general import engine_global_namespace

# NOTE: the ipcluster should be set up before this file is imported
profile = None
dv = []
costs = np.array([])
data_residency = {}
client_ids = []
client = None

def reset_engines():
    global dv
    dv.push(dict(my_data={}))

def setup_engines():
    global client
    global profile
    global dv
    global client_ids
    global data_residency
    global costs
    client = Client(profile=profile)
    dv = client[:]
    client_ids = client._ids
    data_residency = {}
    costs = np.zeros(len(dv))
    reset_engines()

if profile != None:
    profile = profile
    setup_engines()
elif os.environ.has_key("PYHSMM_IPYTHON_PARALLEL_PROFILE"):
    profile = os.environ["PYHSMM_IPYTHON_PARALLEL_PROFILE"]
    setup_engines()

def check_is_ready():
    if profile == None:
        raise RuntimeError("set_profile must be run, e.g., pyhsmm.parallel.set_profile(profile='default')")

def set_profile(this_profile):
    global profile
    profile = this_profile
    os.environ["PYHSMM_IPYTHON_PARALLEL_PROFILE"] = profile
    setup_engines()

def get_num_engines():
    check_is_ready()
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


@engine_global_namespace
def update_my_data(data_id,data):
    my_data[data_id] = data

# interface

def has_data(data):
    return phash(data) in data_residency

def add_data(data,costfunc=len):
    check_is_ready()
    global data_residency
    global costs
    # NOTE: this is basically a one-by-one scatter with an additive parametric
    # cost function treated greedily
    ph = phash(data)
    engine_to_send = np.argmin(costs)
    data_residency[ph] = engine_to_send
    costs[engine_to_send] += costfunc(data)
    idx = client_ids
    return client[idx[engine_to_send]].apply_async(update_my_data,ph,data)

def broadcast_data(data,costfunc=len):
    global data_residency
    global costs
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
    global client
    global dv
    @engine_global_namespace
    def _call(f,data_id,**kwargs):
        return f(my_data[data_id],**kwargs)

    if engine_globals is not None:
        dv.push(engine_globals,block=True)

    if kwargss is None:
        kwargss = [{} for data in added_datas] # no communication overhead

    indata = [(phash(data),data,kwargs) for data,kwargs in zip(added_datas,kwargss)]
    idx = client_ids
    ars = [client[idx[data_residency[data_id]]].apply_async(_call,fn,data_id,**kwargs)
                    for data_id, data, kwargs in indata]
    dv.wait(ars)
    results = [ar.get() for ar in ars]

    client.purge_results('all')
    client.results.clear()
    dv.results.clear()

    return results

def map_on_each_broadcasted(fn,broadcasted_datas,kwargss=None,engine_globals=None):
    raise NotImplementedError # TODO lbv version

def call_with_all(fn,broadcasted_datas,kwargss,engine_globals=None):
    global client
    global dv

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

    client.purge_results('all')
    client.results.clear()
    dv.results.clear()

    return results

