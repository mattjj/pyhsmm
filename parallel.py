from __future__ import division
from IPython.parallel import Client
from IPython.parallel.util import interactive



profile = None

# NOTE: the ipcluster should be set up before this file is imported
from IPython.parallel import Client
c = Client()
dv = c.direct_view()
dv.execute('import pyhsmm')
lbv = c.load_balanced_view()

# this dict needs to be populated by hand before calling build_states*, both
# locally (in this module) and in the ipython top-level module on every engine
# NOTE: the data should probably be arrays with dtype=np.float64
alldata = {}

# this function is run on the engines, and expects the alldata global as well as
# the current model global_model to be present in the ipython global frame
@lbv.parallel(block=True)
@interactive
def build_states(data_id):
    global global_model
    global alldata

    # adding the data to the pushed global model will build a substates object
    # and resample the states given the parameters in the model
    global_model.add_data(alldata[data_id],initialize_from_prior=False)
    stateseq = global_model.states_list[-1].stateseq
    global_model.states_list = []

    return (data_id, stateseq)

# this stuff is for the 'changepoints' models
allchangepoints = {}

@lbv.parallel(block=True)
@interactive
def build_states_changepoints(data_id):
    global global_model
    global alldata, allchangepoints

    global_model.add_data(alldata[data_id],allchangepoints[data_id],initialize_from_prior=False)
    stateseq = global_model.states_list[-1].stateseq
    global_model.states_list = []

    return (data_id, stateseq)
  
@lbv.parallel(block=True)
@interactive
def resample_obs_distns(state):
    global global_model
    global_model.obs_distns[state].resample( ([s.data[s.stateseq == state] for s in global_model.states_list]) )
    return global_model.obs_distns[ state ]

@lbv.parallel(block=True)
@interactive
def resample_states(s):
    global global_model
    global_model.states_list[0].resample()
    return global_model.states_list[0]
  
