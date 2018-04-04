import numpy as np

def v_rad(x,r,inc,mstar):
    '''Return radial velocity in km/s.
    
    Parameters
    ----------
    x : float or ndarray
        Sky distance to pixel along disk major axis.
    r : float or ndarray
        Distance to pixel in m.
    inc : float
        Inclination of disk in degrees.
    mstar : float
        Stellar mass in Solar masses.
    '''
    g, msun = 6.67408e-11, 1.9891e30
    vcirc = np.sqrt(g * msun * mstar / r**3)
    return vcirc * np.sin(np.deg2rad(inc)) * x / 1e3


def stack_vel(c, inclination=None, pa=None, mstar=None, x0=0.0, y0=0.0,
              arcsec_pix=None, distance=None, vel_pix=None, mask=None):
    '''Stack sky/velocity cube for a given inclination.
    
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
        Pixel offset of center from image center.
    y0 : float
        Pixel offset of center from image center.
    arcsec_pix : float
        Pixel scale of (square) image pixels.
    distance : float
        Distance to target in parsec.
    vel_pix : float
        Pixel scale of velocity axis.
    mask : ndarray of bool or int, same shape as c[:,:]
        Mask used to restrict pixels where velocites are stacked.
    '''

    # distances
    sh = c.shape
    x = np.arange(sh[1]) - sh[1]/2 + 0.5 - x0
    y = np.arange(sh[0]) - sh[0]/2 + 0.5 - y0
    x, y = np.meshgrid(x, y)
    r_sky = np.sqrt(x*x + y*y)

    # distance along major axis
    t = np.array([[np.cos(np.deg2rad(pa)),-np.sin(np.deg2rad(pa))],
                  [np.sin(np.deg2rad(pa)), np.cos(np.deg2rad(pa))]])
    XY = np.vstack( (x.reshape(-1), y.reshape(-1)) )
    xy = np.matmul(t.T, XY)
    x, y = xy.reshape( (2,) + x.shape )

    # distance to pixel in plane of orbit
    r = np.sqrt( r_sky*r_sky + (x*np.tan(np.deg2rad(inclination)))**2 )

    # velocities in each pixel
    sc = arcsec_pix * distance * 1.496e11
    vr = v_rad(y * sc, r * sc, inclination, mstar)

    # shift cube by rolling each sky pixel,
    # do this by shifting the x's in each layer of y
    c_shift = np.zeros(sh)
    shift = np.round(vr / vel_pix).astype(int)
    for i in range(sh[0]):
        c_shift[i] = np.roll(c[i], tuple(shift[i]), 1)

    return shift, c_shift
