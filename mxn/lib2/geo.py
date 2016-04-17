"""Geography / geodesy related.

Maybe rename to gis?

"""

import numpy as np

from . import coordinates
#from . import const

def lonlat_deg2meter(latd=0):
    """Returns approximate value of 1 deg of Lon/Lat (WGS84) in meters.

    Input: Latitude in Degrees.
    Expects the Earth ellipsoid constants in const.
    Absolute values are returned.
    """
    return lonlat_rad2meter(np.radians(latd))*np.pi/180

def lonlat_rad2meter(lat=0):
    """Returns approximate value of 1 radian of Lon/Lat (WGS84) in meters.

    Input: Latitude in Radians.
    Expects the Earth ellipsoid constants in const.
    Absolute values are returned.
    """
    phi = lat
    a  = coordinates.WGS84.semimajor_axis
    e2 = coordinates.WGS84.eccentricity_sq

    # The distance of 1 radian Lon/Lat in metres on the WGS84 spheroid
    # is given to within one centimeter by
    # dLatM = 111132.954 - 559.822 * np.cos(2*phi) + 1.175 * np.cos(4*phi)
    # dLonM = np.pi*a*np.cos(phi) / (180 * np.sqrt(1-e2 *(np.sin(phi))**2))
    dLatM = (111132.954 - 559.822 * np.cos(2*phi) + 1.175 * np.cos(4*phi))/np.radians(1)
    dLonM = a*np.cos(phi) / (np.sqrt(1-e2 *(np.sin(phi))**2))
    return np.abs([dLonM, dLatM])

def test_lonlat_res(ref_latd=30.):
    """ref_latd is reference latitude, in degrees"""
    print("1 radian at equator in meters: Lon: {:.1f} km, Lat: {:.1f} km".format(
        *lonlat_rad2meter() /1e3))
    print("Resulting Equator circumference: Lon: {:.1f} km, Lat: {:.1f} km".format(
        *lonlat_rad2meter() *2*np.pi /1e3))
    print("1 degree at equator in meters: Lon: {:.1f} km, Lat: {:.1f} km".format(
        *lonlat_deg2meter() /1e3))
    print("Resulting Equator circumference: Lon: {:.1f} km, Lat: {:.1f} km".format(
        *lonlat_deg2meter() *360 /1e3))

    print("\nAt reference latitude = {:.1f} deg".format(ref_latd))
    print("1 radian in meters: Lon: {:.1f} km, Lat: {:.1f} km".format(
        *lonlat_rad2meter(np.radians(ref_latd)) /1e3))
    print("Resulting circumferences: Lon: {:.1f} km, Lat: {:.1f} km".format(
        *lonlat_rad2meter(np.radians(ref_latd)) *2*np.pi /1e3))
    print("1 degree in meters: Lon: {:.1f} km, Lat: {:.1f} km".format(
        *lonlat_deg2meter(ref_latd) /1e3))
    print("Resulting circumferences: Lon: {:.1f} km, Lat: {:.1f} km".format(
        *lonlat_deg2meter(ref_latd) *360 /1e3))
