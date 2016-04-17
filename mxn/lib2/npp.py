"""NumPy-Plus

Additional convenience functions for efficient working with numpy arrays.

Since it import * from numpy, this module can be used instead of numpy.

"""

import numpy as np
from numpy import *


def np_and(*a):
    """Numpy logical_and over multiple boolean arrays.

    Usage:
    valid = np_and(a, b, c, d)
    where a,b,c,d are boolean arrays with same shape.

    a,b,c,d are 2d max. Generalize later if needed.
    """
    return np.vstack(a).min(axis=0)

def np_or(*a):
    """Numpy logical_and over multiple boolean arrays.

    Usage:
    valid = np_and(a, b, c, d)
    where a,b,c,d are boolean arrays with same shape.

    a,b,c,d are 2d max. Generalize later if needed.
    """
    return np.vstack(a).max(axis=0)


def circ_stats(data, nan=True, deg=False):
    """Returns dict of directional statistics values, as defined in
    http://en.wikipedia.org/wiki/Directional_statistics

    data : array like, complex, or float (assumed the argument of complex)
    nan : bool, either to check for finite
    deg : bool, either to add additional values in degrees (in addition to radians)

    Note: it makes sense to consider the data as normalized or not, in
    dependence of application. The length of vectors is taken into account in
    this case. Otherwise, normalize the data prior to calling the function.

    Processed in blocks, to accomodate very large data sets.
    """
    d = np.asarray(data).ravel() # makes a 1-d np array.
    #bs = 1e6 # blocksize
    if not np.iscomplexobj(d):
        d = np.exp(1j * d)
    if nan:
        valid = np.where(np.isfinite(d))
    else:
        valid = slice(d.size)

    r = np.sum(d[valid])   # direction vector
    R = np.abs(r) / d.size # direction vector length
    circ_var = 1-R
    circ_std = np.sqrt(-2*np.log(R))
    circ_spread = circ_var * 2*np.pi # custom ad-hoc feature
    stats = {'mean_angle': np.angle(r),
             'circ_var': circ_var,
             'circ_std': circ_std,
             'circ_spread': circ_spread,
             'valid': len(valid[0])/d.size*100 if nan else 100., # in %
             }
    # adding values in degrees
    stats['mean_angle_deg'] = np.degrees(stats['mean_angle'])
    stats['circ_spread_deg'] = np.degrees(stats['circ_spread'])
    return stats
