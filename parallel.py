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

### adding and managing data

# internals

_data_to_id_dict = defaultdict(count().next)
_id_to_data_dict = {}
def _data_to_id(data):
    # NOTE: keyed by object memory address, not values in data
    # that's so that the same data array can be added multiple times
    return _data_to_id_dict[data.__hash__()]

def _id_to_data(data_id):
    return _id_to_data_dict[data_id]

def _send_data_to_an_engine(data,costfunc=len):
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

def add_data(data,already_loaded,**kwargs):
    _id_to_data_dict[_data_to_id(data)] = data
    if not already_loaded:
        return _send_data_to_an_engine(data,**kwargs)

def call_data_fn(fn,datas,kwargss=None,engine_globals=None):
    assert all(data in _data_to_id_dict for data in datas)

    if engine_globals is not None:
        dv.push(engine_globals,block=False)

    data_ids_to_resample = set(_data_to_id(data) for data in datas)

    if kwargss is None:
        results = dv.apply_sync(_call_data_fn,fn,data_ids_to_resample)
    else:
        kwargs_for_each_data = {_data_to_id(data):kwargs for data,kwargs in zip(datas,kwargss)}
        results = dv.apply_sync(_call_data_fn,fn,data_ids_to_resample,kwargs_for_each_data)

    c.purge_results('all')
    return [(_id_to_data(data_id),outs) for result in results for data_id, outs in result]


# TODO data-everywhere, dynamic load balancing version
# def broadcast_data(data):
#     return dv.apply(_update_mydata,_data_to_id(data),data)

