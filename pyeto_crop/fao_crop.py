"""
Library of functions for estimating reference evapotransporation (ETo) for
a grass reference crop using the FAO-56 Penman-Monteith and Hargreaves
equations. The library includes numerous functions for estimating missing
meteorological data.

:copyright: (c) 2020 by Ronja Iffland.
:license: BSD 3-Clause, see LICENSE.txt for more details.
"""


import numpy as np


def lat(t):
    """
    Estimate latent heat of vaporization.
    
    Based on equation 3-1 in Allen et al (1998).
    
    :param t: Temperature [deg C]
    :return: latent heat of vaporization [MJ kg-1]
    """ 
    return 2.501 - (2.361 * 10**(-3)) * t


def pa(pres, t):
    """
    density of air
    
    Based on Box 6 in Allen et al (1998).
        
    :param pres: atmospheric pressure [kPa]
    :param t: Air temperature at 2 m height [deg C]
    :return: mean air density at constant pressure [kg m-3]
    """
    
    return pres / ((1.01 * (t + 273)) * 0.287)


def ra(d, zom, zoh, ws, h=0.12, zh=2, zm=10, k=0.41):
    """
    Estimate aerodynamic resistance.
    
    Based on equation 4 in Allen et al (1998).
    
    :param zh: height of humidity measurements [m]
    :param zm: height of wind measurement [m]
    :param d: zero plane displacement height [m]
    :param zom: roughness length governing momentum transfer [m]
    :param zoh: roughness length governing transfer of heat and vapour [m]
    :param k: con Karman's constat = 0.41 [-]
    :param ws: windspeed at zm [m s-1]
    :parm h: crop height [m]
    :return: aerodynamic resistance [s m-1]
    """
    return (np.log((zm - d) / zom) * np.log((zh - d) / zoh)) / (k**(2) * ws)


def d(h=0.12):
    """
    Estimate zero plane displacement height
    
    Based on equation 4 in Allen et al (1998).
    
    :param h: crop height [m]
    :return: zero plane displacement height [m]
    """
    return 2/3 * h


def zom(h=0.12):
    """
    roughness length governing momentum transfer
    
    Based on equation 4 in Allen et al (1998).
    
    :param zom: roughness length governing momentum transfer [m]
    :return: roughness length governing momentum transfer [m]
    """
    return 0.123 * h


def zoh(zom):
    """
    roughness length governing transfer of heat and vapour
    
    Based on equation 4 in Allen et al (1998).
    
    :return: roughness length governing transfer of heat and vapour [m]
    """
    return 0.1 * zom


def rs(LAIactive, rl):
    """"
    Estimate surface resistance.
    
    Based on equation 5 in Allen et al (1998).
      
    :param rl: bulk stomatal reiststance of the well-illuminated leaf [s m-1]
        = 100 under well-watered conditions
    :param LAIactive: active (sunlit) leaf area index
        [m2 (leaf area) m-2 (soil surface)]
    :return: surface resistance [s m-1]
    
    """
    return rl / LAIactive


def LAIactive(LAI):
    """
    active (sunlit) leaf area index [m2 (leaf area) m-2 (soil surface)]
    
    Based on equation 5 in Allen et al (1998).
    
    :param LAI: leaf area index [m2 (leaf area) m-2 (soil surface)]
    :return: active (sunlit) leaf area index
    """
    return 0.5 * LAI

def LAI(h=0.12):
    """
    general equation for clipped grass
    
    :param h: crop height [m]
    :return: LAI
    """
    
    return 24 * h


def penman_monteith(lat, net_rad, t, ws, svp, avp, delta_svp, d, psy, ra, pa, pres, rs=78, cp=0.001013, shf=0.0):
    """
    Estimate evapotranspiration (ETc) from specific crop type using the Penman-Monteith equation.

    Based on equation 3 in Allen et al (1998).

    :param lat: latent heat of vaporization [MJ kg-1].
    :param delta_svp: Slope of saturation vapour pressure curve [kPa deg C-1].
        Can be estimated using ``delta_svp()``.
    :param net_rad: Net radiation at crop surface [MJ m-2 day-1]. If
        necessary this can be estimated using ``net_rad()``.
    :param shf: Soil heat flux (G) [MJ m-2 day-1] (default is 0.0, which is
        reasonable for a daily or 10-day time steps). For monthly time steps
        *shf* can be estimated using ``monthly_soil_heat_flux()`` or
        ``monthly_soil_heat_flux2()``.
    :param pa: density of air [kg m-3]
    :param pres: atmospheric pressure [kPa]
    :param cp: specific heat of moist air [MJ kg-1 C-1] = 1.013 KJ kg-1 C-1 for
        av. atmosphere conditions (Allen 1998)
    :parm: ra: aerodynamic resistance [s m-1].
    :param psy: Psychrometric constant [kPa deg C-1]. Can be estimatred using
        ``psy_const_of_psychrometer()`` or ``psy_const()``.
    :param rs: surface resistence [s m-1].
    :param t: Air temperature at 2 m height [Kelvin].
    :param ws: Wind speed at 2 m height [m s-1]. If not measured at 2m,
        convert using ``wind_speed_at_2m()``.
    :param svp: Saturation vapour pressure [kPa]. Can be estimated using
        ``svp_from_t()''.
    :param avp: Actual vapour pressure [kPa]. Can be estimated using a range
        of functions with names beginning with 'avp_from'.
    :return: Evapotranspiration (ETc) from a specific crop type [mm day-1].
    :rtype: float
    """
    
    a1 = delta_svp * (net_rad - shf)/86400 + pa * cp * ((svp - avp) / (ra))
    a2 = delta_svp + psy * (1 + (rs / ra))
    a3 = 1 / lat
    
    return (a1 / a2) * a3 * 86400
    
