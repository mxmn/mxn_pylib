"""Library of utility/convenience routines and classes.

Includes:
- proc  : data processing (block-processing, image-processing, etc.)
- io    : file/data in/output
- timer : timing/progress related
- const : physical and mathematical constants
- vis   : graphics visualizations
- radar : radar/sar related routines (includes cc, mtv)
- base  : other convenience functions that didn't fit somewhere else
- geo   : geographical routines
- coordinates : geographical coordinate transformations
- npp   : numpy convenience functions

To be included/extended:
- file io for: envi, isce, rat
- more stats

Note:
- The provided reload() function, only reloads the modules. The individual
  classes and functions imported via "from .xx import *" are not reloaded.
  --> if needed, either use full module name paths (lib.radar.mtv()),
      or re-import them manually.

"""

# independent stand-alone modules
from .proc import *
from .io import *
from .timer import *
from .const import *
from .vis import *

# depend on previous modules
from .radar import *
from .base import *

# not imported into "lib" namespace, have their own namespace
from . import coordinates
from . import geo
from . import mbpi
from . import npp


def reload():
    """Reloading all modules of this package.

    importlib.reload() does not redefine objects imported in
    "from .. import" form.
    --> either import the parts directly, or re-execute the from statements.
    """
    import importlib

    importlib.reload(proc)
    importlib.reload(io)
    #importlib.reload(timer) # since timer method available
    importlib.reload(const)
    importlib.reload(vis)
    importlib.reload(radar)
    importlib.reload(base)

    importlib.reload(coordinates)
    importlib.reload(geo)
    importlib.reload(mbpi)
    importlib.reload(npp)
