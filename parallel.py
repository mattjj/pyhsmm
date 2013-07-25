from __future__ import division
from IPython.parallel import Client

# NOTE: the ipcluster should be set up before this file is imported

c = Client()
dv = c.direct_view()
lbv = c.load_balanced_view()

dv.push(dict(alldata={},mydata={}))

def get_num_engines():
    return len(dv)

def is_on_engines(data):
    data.flags.writeable = False
    h = hash(data.data)
    return any(h in hashes for hashes in _get_data_hashes)

@dv.remote(block=True)
def _get_data_hashes():
    for data in mydata.values():
        data.flags.writeable = False
    return [hash(data.data) for data in mydata.values()]



def add_data(data):
    pass # TODO

def send_data_to_an_engine(data,costfunc=lambda x: len(x)):
    pass # TODO

@dv.remote(block=True)
def _get_engine_costs(costfunc):
    return sum(costfunc(d) for d in mydata.values())

def call_resampler(resampler_fn,datas,engine_globals={},kwargs={}):
    dv.push(engine_globals,block=False)
    # TODO map data to ids
    _call_resampler_fn(resampler_fn,**kwargs)
    c.purge_results('all')
    # TODO map ids back to data
    # TODO return

@dv.remote(block=True)
def _call_resampler_fn(f):
    pass


# gotta map hashes to data anyway, or some key to data. might as well be ints!

