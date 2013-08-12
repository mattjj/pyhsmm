# __all__ = something
import models
import basic
import basic.distributions as distributions # shortcut
import plugins
import util

import os
EIGEN_INCLUDE_DIR = os.path.join(os.path.dirname(__file__),'deps/Eigen3')
