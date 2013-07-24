from __future__ import division
from IPython.parallel import Client

# NOTE: the ipcluster should be set up before this file is imported

c = Client()
dv = c.direct_view()
dv.execute('import pyhsmm')
lbv = c.load_balanced_view()

