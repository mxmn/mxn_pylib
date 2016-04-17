"""SAR/Radar related routines and classes.

SAR/Radar related routines assume 2d radar images with multiple channels
with the structure [az, rg, ...].
This can be:
- [az, rg] - single channel data (e.g. single-pol slc)
- [az, rg, 3] - 3 channel data (e.g. 3 polarization channels)
- [az, rg, 2, 3] - 2 tracks with 3 polarizations each (SB-PolInSAR)
- [az, rg, n_tr, n_pol] - multi-baseline PolInSAR scattering vectors
- [az, rg, n_tr*n_pol, n_tr*n_pol] - multi-baseine PolInSAR covariance matrix


Includes:
- db / db2lin
- cc  : complex coherence computation from 2 channes, with optional phase offset.
        Accepts either presumming parameter (div), or smoothing parameter (smm).
        Should handle numpy, memmap, and h5py arrays.
- mtv : convenient visualization function for radar data (and not only).
        Should handle numpy, memmap, and h5py arrays.

Modifications:
- 4/23/15, mn: show_slc_spectrum() added

"""

import numpy as np
import scipy as sp
import pylab as plt

#from mxn.lib.base import normscl
#from mxn.lib.proc import *
from .base import normscl
from .proc import *

def db(x):
    """From linear to decibel"""
    return 10.0*np.log10(x)

def db2lin(x):
    """From decibel to linear"""
    return 10.0**(x/10.0)


def magscale(img, factor=2.5, div=None, type='mag'):
    """Scales radar image magnitude.
    Options:
      - type : {'slc', 'amp', 'mag'}
      - div : when provided, image is shrinked
    """
    if type in ['slc', 'amp','a']:
        func = lambda x: np.abs(x)**2
    elif type in ['mag','m','i']:
        if img.dtype in ['F', 'D']:
            func = np.abs
        else:
            func = None
    if div is not None and func is not None:
        mag = block_rebin(func, img, div=div, dtype='f', bs=div[0]*2)
    elif func is not None:
        mag = block_filter(func, img, dtype='f')
    elif div is not None:
        mag = rebin(img, div=div)
    else:
        mag = img
    n = np.shape(mag)
    if len(n) == 3:
        ret = np.zeros(n,dtype='float32')
        for i in range(n[2]):
            im = mag[:,:,i]
            ret[:,:,i] = np.clip(im/np.mean(im[im > 0])*255//factor,0,255)
        return ret
    return np.clip(mag/np.mean(mag[mag > 0])/factor*255,0,255)

# OBSOLETE:
# def show_slc(a, div=[20,5], type='amp'):
#     """prepares to show slc magnitude"""
#     mag = block_rebin(lambda x: np.abs(x)**2, a, div=div, dtype='f', bs=div[0])
#     return magscale(mag)

def img_ml(img, div=None, smm=None):
    """Multi-look given image, either using presumming and/or smoothing"""
    if (div is not None and np.max(div) > 1):
        res = rebin(img, div=div)
    else:
        res = np.array(img)
    if (smm is not None and np.max(smm) > 1):
        res = smooth(res, smm)
    return res


def cc(s1, s2, ph=None, smm=None, div=None):
    """Complex coherence either by rebin or by smooth.

    ph : array_like, float
        Additional phase argument, e.g. flat-earth or topography.
        Correction: s1*s2*exp(-i*ph)
    """
    only_div = div is not None and (smm is None or np.max(smm) <= 1)
    only_smm = smm is not None and (div is None or np.max(div) <= 1)
    if only_div:
        bs = div[0]
        if ph is None:
            coh = (block_rebin2(lambda a,b: a*np.conj(b),s1,s2,bs=bs,div=div)/
                   np.sqrt(block_rebin(lambda x: np.abs(x)**2,
                                       s1, bs=bs, div=div, dtype=float) *
                           block_rebin(lambda x: np.abs(x)**2,
                                       s2, bs=bs, div=div, dtype=float)))
        else:
            coh = (block_rebin3(
                lambda a,b,c:a*np.conj(b)*np.exp(-1j*c),s1,s2,ph,bs=bs,div=div)
                   /np.sqrt(block_rebin(lambda x: np.abs(x)**2,
                                        s1, bs=bs, div=div, dtype=float) *
                            block_rebin(lambda x: np.abs(x)**2,
                                        s2, bs=bs, div=div, dtype=float)))
    elif only_smm:
        if ph is None:
            coh = (smooth(s1 * np.conj(s2),smm)/
                   np.sqrt(smooth(np.abs(s1)**2,smm)*
                           smooth(np.abs(s2)**2,smm)))
        else:
            coh = (smooth(s1 * np.conj(s2)*np.exp(-1j*np.asarray(ph)),smm)/
                   np.sqrt(smooth(np.abs(s1)**2,smm)*
                           smooth(np.abs(s2)**2,smm)))
    else:
        raise Exception("Not Implemented Yet... (both div/smm or none)")
    return coh


def mtv(d, type='n', div=[10,10], div_adaptive=False,
        # selection of plot features (cb=colorbar)
        figure=True, contour=False, mask=None,
        cb=True, cb_label=None,
        cb_kwargs={}, # colorbar keyword-args
        # standard pylab keywords:
        figsize=(13,11), title=None, origin='lower', reverse_x=False,
        cmap=None, vrange=None,
        dpi = 80, **kwargs):
    """Show image as in idl mtv.
    Supported types: 'm', 'p', 'coh'='c', 'n'=none, etc.
    Use div to reduce image size.
    ! It is usually applied after other operations !
    To improve speed, apply rebin before callling mtv: mtv(rebin(some))

    Other options to imshow:
    - interpolation = "nearest" or "bilinear", or "bicubic", etc.
    - vmin, vmax (alternative: vrange=[vmin, vmax])

    Colorbar keyword args/options (cb_kwargs):
    - shrink : 0.5
    - orientation : {horizontal, vertical}
    - format : { '%.3f' }

    If no presuming desired, set div=[1,1].  If div_adaptive, it will compute
    div to correspond to a given dpi per figsize inch. div=None is equivalent
    to div_adaptive=True.

    Mod Log:
    - 9/25/15: added vrange parameter, shortcut for vrange=[vmin, vmax]
    - 9/25/15: added cb_label parameter
    """
    if div is None or div_adaptive:
        n = np.array(np.shape(d)[:2])
        div = np.maximum(n // (np.array(figsize)[::-1]//2 * dpi), [1,1])
        print("Adaptive image division factors: ",div)
    if 'vmin' in kwargs and 'vmax' in kwargs :
        vrange = [kwargs['vmin'], kwargs['vmax']]


    ismag = type.lower() in ['m','mag','i','intensity','ref','pow','pwr']
    isamp = type.lower() in ['slc','amp','a']
    ispha = type.lower() in ['p','pha','phase']
    iscoh = type.lower() in ['c','coh']
    isdif = type.lower() in ['dif'] # symmetric difference image

    if ispha: # accepts complex values, or real, assuming in radians
        img = np.degrees(rebin(
            np.angle(d) if np.iscomplexobj(d) else d, div=div))
        if cmap is None: cmap = 'hsv'
        if vrange is None: vrange = [-180, 180]
    elif ismag:
        img = magscale(d,div=div,type='mag')
    elif isamp:
        img = magscale(d,div=div,type='amp')
    elif iscoh:
        img = rebin(np.abs(d),div=div)
        if vrange is None: vrange = [0,1]
    else:
        img = rebin(d,div=div)
    if isdif and cmap is None: cmap = 'RdYlGn' # 'bwr'

    n = np.shape(img)
    if len(n) == 3 and n[2] in [3,4]:
        for i in range(n[2]):
            img[:,:,i] = normscl(img[:,:,i])

    if vrange is not None:
        kwargs['vmin'] = vrange[0]
        kwargs['vmax'] = vrange[1]

    #plt.ion()
    if figure:
        plt.figure(figsize=figsize)
    if mask is not None:
        if mask.shape != img.shape:
            mask = sp.ndimage.interpolation.zoom(
                mask, np.array(img.shape)/np.array(mask.shape), order=0)
            #mask = rebin(mask, div=div).astype(np.bool)
        img = np.ma.masked_where(1-mask, img)

    if ismag or isamp:
        if cmap is None: cmap=plt.get_cmap("gray")
        plt.imshow(img,origin=origin,
                   cmap = cmap, **kwargs)
    elif iscoh:
        if cmap is None: cmap=plt.get_cmap("gray")
        plt.imshow(img, origin=origin,
                   cmap=cmap, **kwargs)
    else:
        plt.imshow(img, origin=origin, cmap=cmap, **kwargs)

    if cb:
        if "shrink" not in cb_kwargs:
            cb_kwargs["shrink"] = 0.7
        cbar = plt.colorbar(**cb_kwargs)
        if cb_label is not None:
            cbar.set_label(cb_label)

    if contour:
        cont = plt.contour(img,origin=origin, cmap=plt.get_cmap("gray"),
                           linewidths=2, **kwargs)
        plt.clabel(cont,fmt='%1.1f')

    if reverse_x:
        plt.xlim(reversed(plt.xlim()))


    if title is not None:
        plt.title(title)
    #plt.show()


def show_slc_spectrum(slc, ch_lab=None, div=[3,12], spacing=[1,1],
                      show_freq=True, show_db=False):
    """Show range and azimuth spectra for SLC images.

    slc : np.array or a list of lists: [n_tr, n_pol, az, rg]
    ch_lab : labels for different channels [[tr1, tr2], [hh, vv, hv]]
    div : [az, rg] division window size for visualization and averaging
    spacing : [az, rg] pixel spacing in meters.
    show_freq : either to plot over frequencies, not bins.
    show_db : either to plot in db, not linear.
    """
    from numpy.fft import fft, fft2, fftshift, fftfreq
    import pylab as plt
    plt.ion()
    n = np.shape(slc)
    if len(n) != 4:
        print("Not standard form of a list of lists: [n_tr][n_pol][az, rg]")
        print("Need to convert to numpy array (potentially memory intensive).")
        if len(n) == 2:
            slc = np.array(slc).reshape((1,1,n[0],n[1]))
        if len(n) == 3:
            # Next line changed by MWD - assume that if only two slcs are being
            # shown, they are same pol but diff tracks, for comparison.
            slc = np.array(slc).reshape((n[0],1,n[1],n[2]))
        n = np.shape(slc)

    n_tr = n[0]
    n_pol = n[1]

    if ch_lab is None:
        ch_lab = [["Track "+str(i) for i in range(n_tr)],
                  [["HH","VV","HV"][i] for i in range(n_pol)]]
        if n_pol > 3:
            ch_lab[1] = ["Ch "+str(i) for i in range(n_pol)]

    ffaz = np.array([[np.fft.fft(
        slc[tr][p],axis=0) for p in range(n_pol)] for tr in range(n_tr)])
    ffrg = np.array([[np.fft.fft(
        slc[tr][p],axis=1) for p in range(n_pol)] for tr in range(n_tr)])
    ff2 = np.fft.fft2(slc[0][0])

    dx = np.array(spacing)     # pixel spacing in meters
    extent = n[2:3] * dx       # image extent in meters
    xfaz = fftfreq(n[2],dx[0]) # frequencies in azimuth/y
    xfrg = fftfreq(n[3],dx[1]) # frequencies in range/x
    if show_freq:
        az, rg = xfaz, xfrg
    else:
        az, rg = np.arange(n[2]), np.arange(n[3])

    faz = np.sum(np.abs(ffaz)**2,axis=3)
    frg = np.sum(np.abs(ffrg)**2,axis=2)

    # roll to min frequency for range spectrum.
    frg = np.roll(frg, xfrg.argmin(),axis=-1)
    xfrg = np.roll(xfrg, xfrg.argmin())

    # to plot in db
    if show_db:
        faz = db(faz)
        frg = db(frg)

    fazn = np.sum(faz,axis=1) # averaged over polarizations and normalized
    for i in range(n_tr): fazn[i,:] /= np.max(fazn[i,:])
    fazn = [np.convolve(fftshift(f),np.ones(11)/11,'valid') for f in fazn]
    frgn = np.sum(frg,axis=1) # averaged over polarizations and normalized
    for i in range(n_tr): frgn[i,:] /= np.max(frgn[i,:])
    frgn = [np.convolve(fftshift(f),np.ones(11)/11,'valid') for f in frgn]

    plt.figure(figsize=(17,14.5))
    plt.subplot(331)
    plt.imshow(magscale(ffaz[0,0],type='amp',div=div),
               aspect='auto',cmap=plt.get_cmap("gray"),
               extent=[0,extent[1],az.min(),az.max()])
    plt.title("Azimuth spectrum")
    plt.xlabel("Range [m]")
    plt.ylabel("Azimuth frequency")
    plt.subplot(332)
    plt.imshow(magscale(ffrg[0,0],type='amp',div=div),
               aspect='auto',cmap=plt.get_cmap("gray"),
               extent=[rg.min(),rg.max(),0,extent[0]])
    plt.title("Range spectrum")
    plt.xlabel("Range frequency [1/m]")

    plt.subplot(334)
    for tr in range(n_tr):
        for p in range(n_pol):
            plt.plot(az,(faz[tr,p])/faz.max(),
                     label=ch_lab[0][tr]+" "+ch_lab[1][p])
    # for tr in range(n_tr):
    #     plt.plot(az,fazn/np.max(fazn), label=ch_lab[0][tr])
    plt.title("Azimuth spectrum")
    plt.xlabel("Azimuth frequency [1/m]")
    plt.subplot(335)
    for tr in range(n_tr):
        for p in range(n_pol):
            plt.plot(rg,(frg[tr,p])/frg.max(),
                     label=ch_lab[0][tr]+" "+ch_lab[1][p])
    plt.legend(loc="best")
    plt.title("Range spectrum")
    plt.xlabel("Range frequency [1/m]")

    plt.subplot(333)
    plt.imshow(magscale(ff2,type='amp',div=div),
               aspect='auto',cmap=plt.get_cmap("gray"),
               extent=[rg.min(),rg.max(),az.min(),az.max()])
    plt.title("Total Spectrum "+ch_lab[0][0]+" "+ch_lab[1][0])
    plt.subplot(336)
    plt.imshow(magscale(slc[0][0],type='amp',div=div),
               aspect='auto',cmap=plt.get_cmap("gray"),
               extent=[0,extent[0],0,extent[1]])
    plt.title("SLC Image "+ch_lab[0][0]+" "+ch_lab[1][0])

    plt.subplot(337)
    for tr in range(n_tr):
        plt.plot(fazn[tr]/np.max(fazn), label=ch_lab[0][tr])
    plt.legend(loc="best")
    plt.title("Averaged Azimuth spectrum")
    plt.subplot(338)
    for tr in range(n_tr):
        plt.plot(frgn[tr]/np.max(frgn), label=ch_lab[0][tr])
    plt.legend(loc="best")
    plt.title("Averaged Range spectrum")
