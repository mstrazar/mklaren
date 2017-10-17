from .align import *
from .align_csi import *
from .alignf import *
from .l2krr import *
from .mklaren import *
from .uniform import *

import os
octv = "/usr/local/octave/3.8.0/bin/octave-3.8.0"
if os.path.exists(octv):
    os.environ["OCTAVE_EXECUTABLE"] = octv