"""Coordinate systems and transformations.

All units are in radians and meters.
This module is assumed to be used in combination with NumPy.

Classes
-------
- Ellipsoid
- Peg - Longitude, Latitude, Heading, LocalRadius, in [rad, rad, rad, m]
- LLH - Longitude, Latitude, Height, in [rad, rad, h]
- XYZ - ECEF/XYZ system (Z=North, X=prime meridian, Y), in [m, m, m]
- SCH - Along-track (along Peg-heading), cross-track, and height, in [m, m, m]
- ENU - Local East, North, UP: x, y, z, relative to reference position (Peg)
- (UTM) -

Notes
-----
- LLH are represented as Longitude, Latitidue, and Height
  (right-hand system, x-y-z)
- In this module Longitude and Latitude, as well as all other angles, are
  in radians.
- Heading is from North. Probably in clockwise direction -- Check!
- All functions are based on numpy.
- Transformation methods are usually lower-case names of the coordinate
  classes one wants to transform to. (e.g. LLH.sch() or LLH.xyz(), etc.).

References
----------
.. [1] T. H. Meyer. Introduction to geometrical and physical geodesy:
    foundations of geomatics. ESRI Press, Redlands, California, 2010.

"""

import numpy as np
from numpy import (array, pi, abs, angle, sqrt, sin, cos, tan,
                   arcsin, arctan, arctan2, degrees)
import pyproj


class Ellipsoid:
    """Ellipsoid representation."""
    def __init__ (self, semimajor_axis=None, eccentricity_sq=None,
                  semiminor_axis=None, select='WGS84', name=""):
        """Expects 'semimajor_axis' and ('eccentricity_sq' or
        'semiminor_axis'), or a predefined ellipsoid with 'select'."""
        if semimajor_axis is None:
            if select == 'WGS84':
                semimajor_axis = 6378137.
                eccentricity_sq = 0.00669437999015
                name = select
            else:
                print("Unknown select ellipsoid")
                raise Exception
        if eccentricity_sq is None and semiminor_axis is not None:
            eccentricity_sq = 1. - (semiminor_axis / semimajor_axis)**2
        self.a = self.semimajor_axis = semimajor_axis
        self.e2 = self.eccentricity_sq = eccentricity_sq
        self.e = self.eccentricy = sqrt(self.e2)
        self.b = self.semiminor_axis = self.a * sqrt(1. - self.e2)
        self.f = self.flattening = 1. - sqrt(1. - self.e2)
        self.ep2 = self.ep_squared = self.e2 / (1. - self.e2)
        self.name = name

    def radius_east(self, lat):
        """Radius of curvature in the east direction (lat in radians).
        Also called 'Radius of curvature in the prime vertical' N."""
        return self.a / sqrt(1. - self.e2 * sin(lat)**2)

    def radius_north(self, lat):
        """Radius of curvature in the north direction (lat in radians).
        Also called 'Radius of curvature in the meridian' M."""
        return (self.a*(1.-self.e2) / (1.-self.e2*sin(lat)**2)**1.5)

    def radius_local(self, lat, hdg):
        """Local radius of curvature along heading (lat, hdg in radians)
        Heading is from North (y, lat)!
        It is related to the 'Radius of curvature in the normal section',
        except of the different definition for the hdg/azimuth angle
        (shifted by 90 degrees).
        Direction of heading (+/-) is irrelavant.
        """
        er = self.radius_east(lat)
        nr = self.radius_north(lat)
        return er * nr / (er * cos(hdg)**2 + nr * sin(hdg)**2)


# Default ellipsoid
WGS84 = Ellipsoid(select="WGS84")


class Peg:
    """Peg representation.

    Attributes
    ----------
    lon : radians
        Longitude, in radians
    lat : radians
        Geodetic Latitude, in radians
    hdg : radians
        Heading, in radians, from North, probably in counter-clockwise direction(?).
        (in tdx data it is from North, mathematical direction)
    radius : m
        Local Earth radius, in m
    ellipsoid : Ellipsoid, optional
        Ellipsoid object. Default is WGS-84.
    """

    def __init__(self, lon, lat, hdg, ellipsoid=WGS84):
        self.lon = lon
        self.lat = lat
        self.hdg = hdg
        self.radius = ellipsoid.radius_local(lat, hdg)
        self.ellipsoid = ellipsoid
        self._update_transformations()
    def _update_transformations(self):
        slon, clon = sin(self.lon), cos(self.lon)
        slat, clat = sin(self.lat), cos(self.lat)
        shdg, chdg = sin(self.hdg), cos(self.hdg)
        r = self.radius
        xyzP_to_enu = array([[0, shdg, -chdg],
                             [0, chdg,  shdg],
                             [1,    0,     0]])
        enu_to_xyz = enu_to_xyz_matrix(self.lon, self.lat)
        self.rotation_matrix = enu_to_xyz.dot(xyzP_to_enu)
        re = self.ellipsoid.radius_east(self.lat)
        p = array([re * clat * clon,
                   re * clat * slon,
                   re * (1.0 - self.ellipsoid.e2) * slat])
        up = self.radius * enu_to_xyz[:,2] # just take the third up vector
        self.translation_vector = p - up
    def __call__(self):
        return array([self.lon, self.lat, self.hdg])
    def __repr__(self):
        return("Peg Lon: {:.3f} deg; Lat: {:.3f}; Heading: {:.1f} deg"
               .format(degrees(self.lon),degrees(self.lat),degrees(self.hdg)))


class LLH:
    """Longitude, geodetic Latitude, Height (lon, lat in radians).

    Parameters
    ----------
    lon : float, array_like
        Longitude, in radians([-180, 180])
    lat : float, array_like
        Geodetic latitude, in radians([-90, 90])
    h : float, array_like
        Height above ellipsoid, in meters
    """
    def __init__(self, lon, lat, h=None):
        self.lon = lon
        self.lat = lat
        self.h = h if h is not None else (lon * 0.)
    def __call__(self):
        return array([self.lon, self.lat, self.h])
    def __repr__(self):
        if np.size(self.lon) == 1:
            return("Lon: {:.3f} deg; Lat: {:.3f}; Height: {:.1f} m"
                   .format(degrees(self.lon),degrees(self.lat),self.h))
        else:
            def mmf(a):
                fmt = '{:.3f}'
                return (fmt+' \u00B1'+fmt+' ['+fmt+'..'+fmt+']').format(
                    np.mean(a),np.std(a),np.min(a),np.max(a))
            return("Lon: {} deg\nLat: {} deg\nHeight: {} m".format(
                mmf(degrees(self.lon)),mmf(degrees(self.lat)),mmf(self.h)))
    def __getitem__(self, i):
        assert i>=0 and i<len(self.h), "Array access requires iterable LLH"
        return LLH(self.lon[i], self.lat[i], self.h[i])
    def xyz(self, ellipsoid=WGS84):
        """Transform to ECEF XYZ coordinates."""
        r = ellipsoid.radius_east(self.lat)
        x = (r + self.h) * cos(self.lat) * cos(self.lon)
        y = (r + self.h) * cos(self.lat) * sin(self.lon)
        z = (r * (1. - ellipsoid.e2) + self.h) * sin(self.lat)
        return XYZ(x, y, z, ellipsoid)
    def enu(self, o_xyz=None, o_llh=None, ellipsoid=WGS84):
        """Transform to ENU, given ENU origin point o."""
        if o_xyz is not None: ellipsoid = o_xyz.ellipsoid
        return self.xyz(ellipsoid).enu(o_xyz=o_xyz,o_llh=o_llh)
    def sch(self, peg):
        """Transform to SCH coordinates, given Peg."""
        return self.xyz(peg.ellipsoid).sch(peg)
    def utm(self, zone):
        """Temporary implementation: uses pyproj for transformations."""
        return UTM(zone=zone).from_llh(self)
    def distance(self, other, radius=WGS84.semimajor_axis):
        """Returns distance in meters between this and another LLH point.
        Using haversine formulat to compute the great-circle distance.

        For more accuracy, the radius can be adapted.

        var R = 6371000; // metres
        var \phi 1 = lat1.toRadians();
        var \phi 2 = lat2.toRadians();
        var \Delta\phi = (lat2-lat1).toRadians();
        var \Delta\lambda = (lon2-lon1).toRadians();
        var a = Math.sin(\Delta\phi/2) * Math.sin(\Delta\phi/2) +
                Math.cos(\phi1) * Math.cos(\phi2) *
                Math.sin(\Delta \lambda /2) * Math.sin(\Delta \lambda /2);
        var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        var d = R * c;
        """
        dlat = other.lat-self.lat
        dlon = other.lon-self.lon
        a = np.sin(dlat/2)**2 + np.cos(self.lat)*np.cos(other.lat)*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = c * (radius + self.h)
        return d
    def distance_small(self, other, radius=WGS84.semimajor_axis):
        """Approximate solution, usint Pythagora's theorem (only for small distances)
        var x = (\lambda 2-\lambda 1) * Math.cos((\phi1+\phi2)/2);
        var y = (\phi2-\phi1);
        var d = Math.sqrt(x*x + y*y) * R;
        """
        x = (other.lon-self.lon) * np.cos((self.lat+other.lat)/2)
        y = other.lat-self.lat
        d = np.sqrt(x**2 + y**2) * (radius + self.h)
        return d



class XYZ:
    """ECEF XYZ cartesian geocentric coordinates.

    Parameters
    ----------
    x : float, array_like
        In direction of prime meridian (lon=0, lat=0).
    y : float, array_like
        In direction lon=90, lat=0
    z : float, array_like
        Close to the rotation axis, with direction North (lat=90).
    """
    def __init__(self, x, y=None, z=None, ellipsoid=WGS84):
        if np.iterable(x) and len(x) == 3 and y is None and z is None:
            self.x = x[0]
            self.y = x[1]
            self.z = x[2]
        else:
            self.x = x
            self.y = y
            self.z = z
        self.ellipsoid = ellipsoid
    def __repr__(self):
        return("x: {} y: {} z: {}".format(*self()))
    def __call__(self):
        return array([self.x, self.y, self.z])
    def __getitem__(self, i):
        assert i>=0 and i<len(self.x), "Array access requires iterable XYZ"
        return XYZ(self.x[i], self.y[i], self.z[i], self.ellipsoid)
    def __add__(self, v):
        if len(v) == 3:
            new = XYZ(self.x, self.y, self.z, ellipsoid=self.ellipsoid)
            new.x += v[0]
            new.y += v[1]
            new.z += v[2]
            return new
        else:
            raise Exception("Wrong input in addition to XYZ.")
    def sch(self, peg):
        """Transform to SCH coordinates, given Peg."""
        if not np.iterable(self.x):
            xyzP = peg.rotation_matrix.T.dot(
                array([self.x,self.y,self.z])-peg.translation_vector)
        else:
            xyzP = peg.rotation_matrix.T.dot(
                array([self.x-peg.translation_vector[0],
                       self.y-peg.translation_vector[1],
                       self.z-peg.translation_vector[2]]))
        r = np.linalg.norm(xyzP, axis=0)
        h = r - peg.radius
        c = peg.radius * arcsin(xyzP[2] / r)
        s = peg.radius * arctan2(xyzP[1], xyzP[0])
        return SCH(peg, s, c, h)
    def llh(self):
        lon = arctan2(self.y, self.x)
        pr = sqrt(self.x**2 + self.y**2) # projected radius
        alpha = arctan(self.z / (pr * sqrt(1.-self.ellipsoid.e2)))
        lat = arctan(
            (self.z + self.ellipsoid.ep2 * self.ellipsoid.b * sin(alpha)**3)
            /(pr - self.ellipsoid.e2 * self.ellipsoid.a * cos(alpha)**3))
        h = pr / cos(lat) - self.ellipsoid.radius_east(lat)
        return LLH(lon,lat,h)
    def enu(self, o_xyz=None, o_llh=None):
        """Transform to ENU, given ENU origin point o (in XYZ!).
        At least o_xyz or o_llh have to be provided!"""
        if o_llh is None: o_llh = o_xyz.llh()
        if o_xyz is None: o_xyz = o_llh.xyz(ellipsoid=self.ellipsoid)
        enu_to_xyz = enu_to_xyz_matrix(o_llh.lon, o_llh.lat)
        return ENU(*enu_to_xyz.T.dot(self()-o_xyz()),o_llh=o_llh,o_xyz=o_xyz)


class ENU:
    """ENU cartesian coordinate system.

    East-North-Up (ENU) coordinate system: cartesian similar to XYZ, just
    translated to origin point O, and rotated to align with the ENU axes.

    Parameters
    ----------
    e : float
    n : float
    u : float
    o_llh : LLH
        Origin given in LLH.
    o_xyz : XYZ
        Origin given in XYZ.
    """
    def __init__(self, e, n, u, o_llh=None, o_xyz=None, ellipsoid=WGS84):
        """At least one of the origin points, o_llh and o_xyz,
        have to be provided."""
        self.e = e
        self.n = n
        self.u = u
        if o_llh is None: o_llh = o_xyz.llh()
        if o_xyz is None: o_xyz = o_llh.xyz(ellipsoid)
        self.o_llh = o_llh
        self.o_xyz = o_xyz
    def __repr__(self):
        if len(self.e) == 1:
            return("ENU e: {} n: {} u: {}".format(*self()))
    def __call__(self):
        return array([self.e, self.n, self.u])
    def __add__(self, v):
        if len(v) == 3:
            new = ENU(self.e, self.n, self.u, o_llh=self.o_llh,
                      o_xyz=self.o_xyz)
            new.e += v[0]
            new.n += v[1]
            new.u += v[2]
            return new
        else:
            raise Exception("Wrong input in addition to ENU.")
    def xyz(self):
        enu_to_xyz = enu_to_xyz_matrix(self.o_llh.lon, self.o_llh.lat)
        return XYZ(*enu_to_xyz.dot(self())+self.o_xyz(),
                   ellipsoid=self.o_xyz.ellipsoid)
    def llh(self):
        return self.xyz().llh()


class SCH:
    """Radar-related spherical coordinate system.

    It is referenced to the Peg position, determining the s and c directions,
    and height values.

    Parameters
    ----------
    peg: Peg
        Peg object, determing the directions of s and c coordinates.
    s : float, array_like
        Along-track curved distance, at the ground, in m.
    c : float, array_like
        Cross-track curved distance, at the ground, in m. Positive is left of s.
    h : float, array_like
        Height above peg sphere (?), in m.

    """
    def __init__(self, peg, s=None, c=None, h=None):
        self.peg = peg
        self.s = s
        self.c = c
        self.h = h
    def __repr__(self):
        return("s: {} c: {} h: {}".format(*self()))
    def __call__(self):
        return array([self.s, self.c, self.h])
    def __getitem__(self, i):
        assert i>=0 and i<len(self.s), "Array access requires iterable SCH"
        return SCH(self.peg, self.s[i], self.c[i], self.h[i])
    def llh(self):
        """Transform to LLH coordinates."""
        return self.xyz().llh()
    def xyz(self):
        """Transform SCH point to XYZ ECEF point."""
        c_angle = self.c / self.peg.radius
        s_angle = self.s / self.peg.radius
        r = self.peg.radius + self.h
        # from spherical to cartesian
        xyz_local = array ([r * cos (c_angle) * cos (s_angle),
                            r * cos (c_angle) * sin (s_angle),
                            r * sin (c_angle)])
        # from local xyz to ECEF xyz
        xyz = self.peg.rotation_matrix.dot(xyz_local) + self.peg.translation_vector
        return XYZ(xyz[0], xyz[1], xyz[2], self.peg.ellipsoid)


class LookVectorSCH(SCH):
    """Geometry of a look vector given in SCH coordinates.

    Meant only as reference. This look vector is given from platform to target:
    lv = sch_target - sch_platform

    Note: by default, the full 3d, non-normalized, vector is considered. Either
    project to incidence plane individually or use appropriate methods.

    Parameterizing the look vector in SCH coordinates:
             [ sin(inc_l) sin(az) ]     [ S ]
    \hat\l = [ sin(inc_l) cos(az) ]  =  [ C ]
             [    -cos(inc_l)     ]     [ H ]
    """
    def __init__(self, sch):
        SCH.__init__(self, sch.peg, sch.s, sch.c, sch.h)
    def range(self):
        if "r" not in self.__dict__:
            self.r = np.linalg.norm(self())
        return self.r
    def incidence_plane_look_angle(self):
        return np.arccos(-self.h/r)


# basic functions
def enu_to_xyz_matrix(lon, lat):
    """ENU to XYZ rotation matrix.

    As well the rotation matrix from sch_hat to xyz_prime (with lon=S, lat=C,
    both S and C in radians, i.e. s/r, and c/r).

    """
    slon, clon = sin(lon), cos(lon)
    slat, clat = sin(lat), cos(lat)
    enu_to_xyz = array([[-slon, -slat * clon, clat * clon],
                        [ clon, -slat * slon, clat * slon],
                        [ 0,     clat,        slat       ]])
    return enu_to_xyz

# temporary solution to convert UTM <-> LonLat
class UTM:
    def __init__(self, x=None, y=None, z=None, zone=None,
                 ellipsoid=WGS84):
        """Set up UTM projection, and if given, save point(s).

        Note: zone is required!

        """
        assert zone is not None, "zone is required in UTM object"
        assert ellipsoid == WGS84, "currently it works only with WGS84"

        self.proj = pyproj.Proj(proj='utm',zone=zone,ellps='WGS84')
        self.x, self.y, self.z = x, y, z
    def __repr__(self):
        return("UTM x: {:.3f} m; y: {:.3f} m; z: {:.1f} m"
               .format(self.x, self.y, self.z))
    def llh(self):
        ll = self.proj(self.x, self.y, inverse=True, radians=True)
        return LLH(ll[0], ll[1], self.z)
    def from_llh(self, llh):
        x = self.proj(llh.lon, llh.lat, radians=True)
        self.x, self.y, self.z = x[0], x[1], llh.h
        return self
