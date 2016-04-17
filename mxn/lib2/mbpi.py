"""Multi-Baseline Polarimetric Interferometric Library.

Lower-level functions & classes for working with MB-PolInSAR covariance matrices.

Conventions:
Covariance matrix data dimensions: [az, rg, pol*tr, pol*tr]
Baseline data dimensions: [az, rg, bl, pol]
Tracks data dimensions: [tr, pol, az, rg], or [az, rg, tr, pol]

Cov matrix indices: baselines considered for i<j, with cov[i,j]
;;;  [ T11  0   1   3   6   10  ... ]
;;;  [     T22  2   4   7   11  ... ]
;;;  [         T33  5   8   12  ... ]
;;;  [             T44  9   13  ... ]
;;;  [                 T55  14  ... ]
;;;  [                     T66  ... ]

All baseline and track and polarization indices start with 0.

"""

import numpy as np
import pylab as plt
import scipy as sp

#
# MB functions
#

def mb_n_bl(n_tr):
    """Returns number of baselines, based on given number of tracks"""
    return n_tr * (n_tr-1) / 2

def mb_tr_ind(bl):
    """Returns tuple of track indices for a specified baseline.

    Based on the triangular number calculations.
    """
    j = np.floor((1+np.sqrt(1+8*(bl)))/2).astype(int)
    i = (bl - j*(j-1)/2).astype(int)
    return (i,j)

def mb_bl_ind(tr1, tr2):
    """Returns the baseline index for given track indices.

    By convention, tr1 < tr2. Otherwise, a warning is printed,
    and same baseline returned.
    """
    if tr1 == tr2:
        print("ERROR: no baseline between same tracks")
        return None
    if tr1 > tr2:
        print("WARNING: tr1 exepcted < than tr2")
    mx = max(tr1, tr2)
    bl = mx*(mx-1)/2 + min(tr1, tr2)
    return bl.astype(int)

def mb_cov_ind(bl, pol=0, pol2=None, n_pol=3):
    """Returns i,j covariance matrix indices for given baseline and pol index.

    If pol2 is not given, then same polarization is assumed (e.g. HH-HH, vs. HH-VV)
    """
    t1,t2 = mb_tr_ind(bl)
    p1,p2 = pol, pol if pol2 is None else pol2

    return (t1*n_pol + p1, t2*n_pol + p2)

def mb_sb(c, bl=0, n_pol=3):
    """Extract a single-baseline covariance from a mb dataset"""
    return c
    pass

#
# CUC functions
#

def cuc_show_bl(M,bl=0):
    """Show CUC for a single pixel for a single baseline"""
    c = mb_sb(M, bl)
    plt.figure(figsize=(6,6))
    # plot coh
