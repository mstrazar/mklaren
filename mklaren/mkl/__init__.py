import os
octv = "/usr/local/octave/3.8.0/bin/octave-3.8.0"
if os.path.exists(octv):
    os.environ["OCTAVE_EXECUTABLE"] = octv