from __future__ import division
import numpy as np
from collections import defaultdict
from itertools import count
from IPython.parallel import Client

from util.general import engine_global_namespace

# NOTE: the ipcluster should be set up before this file is imported

### setup

c = Client()
dv = c[:]
# lbv = c.load_balanced_view()

dv.push(dict(mydata={}))

### util

def parallel_hash(d):
    # NOTE: hash value is based on object addresses in memory on the controller,
    # not data values.
    if isinstance(d,np.ndarray):
        return d.__hash__()
    else:
        return hash(tuple(map(parallel_hash,d)))

### adding and managing data

# internals

_data_to_id_dict = defaultdict(count().next)
_id_to_data_dict = {}
def _data_to_id(data):
    return _data_to_id_dict[parallel_hash(data)]

def _id_to_data(data_id):
    return _id_to_data_dict[data_id]

def _send_data_to_an_engine(data,costfunc):
    # NOTE: this is basically a one-by-one scatter with an additive parametric
    # cost function treated greedily
    engine_to_send = np.argmin(_get_engine_costs(costfunc))
    return c[engine_to_send].apply(_update_mydata,_data_to_id(data),data)

@dv.remote(block=True)
@engine_global_namespace
def _get_engine_costs(costfunc):
    return sum([costfunc(d) for d in mydata.values()])

@engine_global_namespace
def _update_mydata(data_id,data):
    mydata[data_id] = data

@engine_global_namespace
def _call_data_fn(f,data_ids_to_resample,kwargs_for_each_data=None):
    if kwargs_for_each_data is None:
        return [(data_id,f(data))
                for data_id, data in mydata.iteritems() if data_id in data_ids_to_resample]
    else:
        return [(data_id,f(data,**kwargs_for_each_data[data_id]))
                for data_id, data in mydata.iteritems() if data_id in data_ids_to_resample]

# interface

def add_data(data,already_loaded=False,costfunc=len,**kwargs):
    if not already_loaded:
        _id_to_data_dict[_data_to_id(data)] = data
        return _send_data_to_an_engine(data,costfunc=costfunc,**kwargs)
    else:
        # find data on engines (maybe its name should be passed in?), register
        # it with a global id
        raise NotImplementedError

def call_data_fn(fn,datas,kwargss=None,engine_globals=None):
    assert all(parallel_hash(data) in _data_to_id_dict for data in datas)
    # assert all(data_exists_on_engine(data) for data in datas) # assumes ndarray

    if engine_globals is not None:
        dv.push(engine_globals,block=False)

    data_ids_to_resample = set(_data_to_id(data) for data in datas)

    if kwargss is None:
        results = dv.apply_sync(_call_data_fn,fn,data_ids_to_resample)
    else:
        kwargs_for_each_data = {_data_to_id(data):kwargs for data,kwargs in zip(datas,kwargss)}
        results = dv.apply_sync(_call_data_fn,fn,data_ids_to_resample,kwargs_for_each_data)
    c.purge_results('all')
    results = filter(lambda r: len(r) > 0, results)

    assert set(data_ids_to_resample) == \
            set(data_id for result in results for data_id,_ in result), \
            "some data did not exist on any engine"

    return [(_id_to_data(data_id),outs) for result in results for data_id, outs in result]


# TODO data-everywhere, dynamic load balancing version
# def broadcast_data(data):
#     return dv.apply(_update_mydata,_data_to_id(data),data)

