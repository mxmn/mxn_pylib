"""Common constants

Most constants are imported from scipy, which includes constants from math.


Currently accessible
====================

pi                 Pi
golden             Golden ratio
c                  speed of light in vacuum
mu_0               the magnetic constant :math:`\mu_0`
epsilon_0          the electric constant (vacuum permittivity)
G                  Newtonian constant of gravitation
g                  standard acceleration of gravity
k = Boltzman       Boltzmann constant

degree             degree in radians
arcmin             arc minute in radians
arcsec             arc second in radians

minute             one minute in seconds
hour               one hour in seconds
day                one day in seconds
week               one week in seconds
year               one year (365 days) in seconds
Julian_year        one Julian year (365.25 days) in seconds

inch               one inch in meters
foot               one foot in meters
yard               one yard in meters
mile               one mile in meters
light_year         one light year in meters
nautical_mile      one nautical mile in meters

hectare            one hectare in square meters
acre               one acre in square meters

kmh                kilometers per hour in meters per second
mph                miles per hour in meters per second

zero_Celsius       zero of Celsius scale in Kelvin
degree_Fahrenheit  one Fahrenheit (only differences) in Kelvins

(WGS84              WGS84 ellipsoid )
re = earth_radius  radius of the Earth at equator = semimajor axis

dtor = DTOR        degrees to radians factor
radeg = RADEG      radians to degrees factor

"""


from scipy.constants import pi, golden, c, mu_0, epsilon_0, G, g, k, Boltzmann
from scipy.constants import degree, arcmin, arcsec
from scipy.constants import minute, hour, day, week, year, Julian_year
from scipy.constants import inch, foot, yard, mile, light_year, nautical_mile
from scipy.constants import hectare, acre, kmh, mph
from scipy.constants import zero_Celsius, degree_Fahrenheit


earth_radius = 6378137.0

# radians-degrees conversion (based on IDL constants)
dtor = DTOR  = pi / 180  # 0.0174532925199433
radeg = RADEG = 180 / pi  # 57.2957795130823230
