import numpy as np

def v_rad(x,r,inc,mstar):
    '''Return radial velocity in an inclined disk in km/s.
    
    Convention is that inclination in 0 to 90deg for +ve x gives a
    negative velocity, which is towards us.

    Parameters
    ----------
    x : float or ndarray
        Sky distance to pixel along disk major axis, in m.
    r : float or ndarray
        Distance to pixel in m.
    inc : float
        Inclination of disk in degrees.
    mstar : float
        Stellar mass in Solar masses.
    '''
    g, msun = 6.67408e-11, 1.9891e30
    vcirc = np.sqrt(g * msun * mstar / r**3)
    return -vcirc * np.sin(np.deg2rad(inc)) * x / 1e3


def stack_vel(c, inclination=None, pa=None, mstar=None, x0=0.0, y0=0.0,
              arcsec_pix=None, distance=None, vel_pix=None, vff_vc=0.0):
    '''Stack sky/velocity cube for a given inclination.
    
    Assume cube velocity dimension is increasing in rv, so a negative
    velocity requires a positive shift to correct.

    .. todo: implement mask keyword
    
    Parameters
    ----------
    c : ndarray
        Cube of dimension y, x, v_rad.
    inclination : float
        Inclination to compute radial velocities in degrees.
    pa : float
        Position angle of disk in degrees.
    mstar : float
        Stellar mass in Solar masses.
    x0 : float
        Offset of disk center from image center, in arcsec.
    y0 : float
        Offset of disk center from image center, in arcsec.
    arcsec_pix : float
        Pixel scale of (square) image pixels.
    distance : float
        Distance to target in parsec.
    vel_pix : float
        Pixel scale of velocity axis.
    mask : ndarray of bool or int, same shape as c[:,:]
        Mask used to restrict pixels where velocites are stacked.
    vff_vc : float, optional
        Add "free-fall" (towards/away from star) component, fraction of V_Kep. 
    '''

    # distances in pixels
    sh = c.shape
    x = np.arange(sh[1]) - sh[1]/2 + 0.5 - x0 / arcsec_pix
    y = np.arange(sh[0]) - sh[0]/2 + 0.5 - y0 / arcsec_pix
    x, y = np.meshgrid(x, y)
    r_sky = np.sqrt(x*x + y*y)

    # along major axis
    t = np.array([[np.cos(np.deg2rad(pa)),-np.sin(np.deg2rad(pa))],
                  [np.sin(np.deg2rad(pa)), np.cos(np.deg2rad(pa))]])
    XY = np.vstack( (x.reshape(-1), y.reshape(-1)) )
    xy = np.matmul(t.T, XY)
    x, y = xy.reshape( (2,) + x.shape )

    # to pixel in plane of orbit
    r = np.sqrt( r_sky*r_sky + (x*np.tan(np.deg2rad(inclination)))**2 )

    # velocities in each pixel
    sc = arcsec_pix * distance * 1.496e11
    vr = v_rad(y * sc, r * sc, inclination, mstar)

    # optional "free-fall" (+ve is towards star) component
    if vff_vc != 0.0:
        phi = np.arctan2(x, y)
        g, msun = 6.67408e-11, 1.9891e30
        vcirc = np.sqrt(g * msun * mstar / (r*sc)**3) / 1e3
        v_ff = - vff_vc * vcirc * r*sc * np.sin(phi)
        vr += v_ff

    # shift cube by rolling each sky pixel
    c_shift = np.zeros(sh)
    shift = -np.round(vr / vel_pix).astype(int)
    for i in range(sh[0]):
        for j in range(sh[1]):
            c_shift[i,j,:] = np.roll(c[i,j,:], shift[i,j])

    return shift, c_shift
