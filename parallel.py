from __future__ import division
import numpy as np
from collections import defaultdict
from itertools import count
from IPython.parallel import Client

# NOTE: the ipcluster should be set up before this file is imported

### setup

c = Client()
dv = c.direct_view()
lbv = c.load_balanced_view()

dv.push(dict(mydata={}))

### util

def get_num_engines():
    return len(dv)

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

@dv.remote(block=True)
def _get_engine_costs(costfunc):
    return sum([costfunc(d) for d in mydata.values()])

def _update_mydata(data_id,data):
    mydata[data_id] = data

@dv.remote(block=True)
def _call_resampler_fn(f,data_ids_to_resample,**kwargs):
    return [(data_id,f(data)) for data_id, data in mydata.iteritems()
            if data_id in data_ids_to_resample]

# interface

def add_data(data,already_loaded,**kwargs):
    _id_to_data_dict[_data_to_id(data)] = data
    if not already_loaded:
        send_data_to_an_engine(data,**kwargs)

def send_data_to_an_engine(data,costfunc=lambda x: len(x)):
    engine_to_send = np.argmin(_get_engine_costs(costfunc))
    c[engine_to_send].apply(_update_mydata,_data_to_id(data),data)

def call_resampler(resampler_fn,datas,engine_globals={},kwargs={}):
    dv.push(engine_globals,block=False)
    data_ids_to_resample = set(_data_to_id(data) for data in datas)
    assert all(data_id in _id_to_data_dict for data_id in data_ids_to_resample)
    results = _call_resampler_fn(resampler_fn,data_ids_to_resample,**kwargs)
    c.purge_results('all')
    return [(_id_to_data(data_id),outs) for result in results for data_id, outs in result]

