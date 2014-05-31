from __future__ import division
import numpy as np
from IPython.parallel import Client
import os
from util.general import engine_global_namespace

from warnings import warn
warn("This code hasn't been tested in a while...") # TODO

# these globals get set, named here for clarity
client = None
dv = None
costs = None
data_residency = None

def reset_engines():
    global costs, data_residency
    dv.push(dict(my_data={}))
    costs = np.zeros(len(dv))
    data_residency = {}

def set_up_engines():
    global client, dv
    try:
        profile = os.environ["PYHSMM_IPYTHON_PARALLEL_PROFILE"]
    except KeyError:
        profile = 'default'
    if client is None:
        client = Client(profile=profile)
        dv = client[:]
        reset_engines()

def set_profile(this_profile):
    global profile, client
    profile = this_profile
    client = None
    os.environ["PYHSMM_IPYTHON_PARALLEL_PROFILE"] = profile

def get_num_engines():
    return len(dv)

def phash(d):
    'hash based on object address in memory, not data values'
    assert isinstance(d,(np.ndarray,tuple))
    if isinstance(d,np.ndarray):
        return d.__hash__()
    else:
        return hash(tuple(map(phash,d)))

def vhash(d):
    'hash based on data values'
    assert isinstance(d,(np.ndarray,tuple))
    if isinstance(d,np.ndarray):
        d.flags.writeable = False
        return hash(d.data)
    else:
        return hash(tuple(map(vhash,d)))

def clear_ipython_caches():
    client.purge_results('all')
    client.results.clear()
    dv.results.clear()

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
    global data_residency, costs
    set_up_engines()
    # NOTE: this is basically a one-by-one scatter with an additive parametric
    # cost function treated greedily
    ph = phash(data)
    engine_to_send = np.argmin(costs)
    data_residency[ph] = engine_to_send
    costs[engine_to_send] += costfunc(data)
    return client[client._ids[engine_to_send]].apply_async(update_my_data,ph,data)

def broadcast_data(data,costfunc=len):
    global data_residency, costs
    set_up_engines()
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
    global client, dv
    set_up_engines()
    @engine_global_namespace
    def _call(f,data_id,**kwargs):
        return f(my_data[data_id],**kwargs)

    if engine_globals is not None:
        dv.push(engine_globals,block=True)

    if kwargss is None:
        kwargss = [{} for data in added_datas] # no communication overhead

    indata = [(phash(data),data,kwargs) for data,kwargs in zip(added_datas,kwargss)]
    ars = [client[client._ids[data_residency[data_id]]].apply_async(_call,fn,data_id,**kwargs)
                    for data_id, data, kwargs in indata]
    dv.wait(ars)
    results = [ar.get() for ar in ars]

    clear_ipython_caches()

    return results

def map_on_each_broadcasted(fn,broadcasted_datas,kwargss=None,engine_globals=None):
    raise NotImplementedError # TODO lbv version

def call_with_all(fn,broadcasted_datas,kwargss,engine_globals=None):
    global client, dv
    set_up_engines()

    # one call for each element of kwargss
    @engine_global_namespace
    def _call(f,data_ids,kwargs):
        return f([my_data[data_id] for data_id in data_ids],**kwargs)

    if engine_globals is not None:
        dv.push(engine_globals,block=True)

    results = dv.map_sync(
            _call,
            [fn]*len(kwargss),
            [[phash(data) for data in broadcasted_datas]]*len(kwargss),
            kwargss)

    clear_ipython_caches()

    return results


### MISC / TEMP

def _get_stats(model,grp):
    datas, kwargss = zip(*grp)

    mb_states_list = []
    for data, kwargs in zip(datas,kwargss):
        model.add_data(data,stateseq=np.empty(data.shape[0]),**kwargs)
        mb_states_list.append(model.states_list.pop())

    for s in mb_states_list:
        s.meanfieldupdate()

    return [s.all_expected_stats for s in mb_states_list]

