from __future__ import print_function

from functools import lru_cache
import numpy as np
import scipy.interpolate

from . import cube

def rotate_zxz(z1,x,z2):
    '''Return a rotation matrix.
        
    Parameters
    ----------
    z1 : float
        Angle for first z rotation, in degrees.
    x : float
        Angle for x rotation, in degrees.
    z2 : float
        Angle for second z rotation, in degrees.
    '''
    c0 = np.cos(np.deg2rad(z1))
    s0 = np.sin(np.deg2rad(z1))
    t0 = np.array([ [c0,-s0,0], [s0,c0,0], [0,0,1] ])

    c1 = np.cos(np.deg2rad(x))
    s1 = np.sin(np.deg2rad(x))
    t1 = np.array([ [1,0,0], [0,c1,-s1], [0,s1,c1] ])

    c2 = np.cos(np.deg2rad(z2))
    s2 = np.sin(np.deg2rad(z2))
    t2 = np.array([ [c2,-s2,0], [s2,c2,0], [0,0,1] ])

    return np.matmul(t2, np.matmul(t1, t0) )

    '''
    the following could be used in los_image_full, but is slow

        # matrix method, invert (transpose) matrix for X -> x
        # this is much slower than doing it bit by bit below
        t = rotate_zxz(anom, inc, pos+90.0)
        x, y, z = np.meshgrid(self.x-x0, self.y-y0, self.z, indexing='ij')
        XYZ = np.vstack( (x.reshape(-1), y.reshape(-1), z.reshape(-1)) )
        xyz = np.matmul(t.T, XYZ)
        x, y, z = xyz.reshape( (3,) + x.shape )

        rxy2 = x**2 + y**2
        rxy = np.sqrt(rxy2)
        r = np.sqrt(rxy2 + z**2) * self.arcsec_pix
        az = np.arctan2(y, x)
        el = np.arctan2(z, rxy)

        # the density, dimensions are x, y, z -> y, x, z
        img_cube = self.emit(r, p[slice(6,6+self.n_emit_params)]) * \
                   self.dens(r, az, el, p[6+self.n_emit_params:])
        img_cube = np.rollaxis(img_cube, 1)

        if cube:
            return img_cube
        else:
            image = np.zeros((self.ny, self.nx))
            image[self.ny2-self.rmax[1]:self.ny2+self.rmax[1],
                  self.nx2-self.rmax[0]:self.nx2+self.rmax[0]] = np.sum(img_cube,axis=2)
            return image
    '''

def convmf(m_in, e_in):
    """Convert array of mean to true anomaly (for single e).

    From Vallado

    .. todo: tidy and include other orbit cases
    """

    m = np.array(m_in) % (2. * np.pi)
    numiter = 50
    small = 0.00000001
    if e_in > small:

        ecc = e_in * 1.0

        #       ;; /* ------------  initial guess ------------- */
        e0 = m + ecc
        lo = np.logical_or( (m < 0.0) & (m > -np.pi), m > np.pi)
        e0[lo] = m[lo] - ecc

        ktr = 1
        e1  = e0 + (m - e0 + ecc * np.sin(e0)) / (1.0 - ecc * np.cos(e0))
        while (np.max(np.abs(e1 - e0)) > small) & (ktr <= numiter):
            ktr += 1
            do = np.abs(e1 - e0) > small
            e0[do] = e1[do]
            e1[do] = e0[do] + (m[do] - e0[do] + ecc * np.sin(e0[do])) / (1.0 - ecc * np.cos(e0[do]))

        #       ;; /* ---------  find true anomaly  ----------- */
        sinv = (np.sqrt(1.0 - ecc * ecc) * np.sin(e1)) / (1.0-ecc * np.cos(e1))
        cosv = (np.cos(e1) - ecc) / (1.0 - ecc * np.cos(e1))
        nu   = np.arctan2( sinv, cosv)

    else:
        #       ;; /* --------------------- circular --------------------- */
        ktr = 0
        nu  = m
        e0  = m

    if ktr > numiter:
        print('WARNING: convmf did not converge')

    return nu


@lru_cache(maxsize=2)
def convmf_lookup(n=200):
    '''Return interpolation object for convmf.'''
    Ms = np.linspace(-np.pi, np.pi, n)
    es = np.linspace(0, 1, n)
    f = np.zeros((n,n))
    for i,m in enumerate(Ms):
        for j,e in enumerate(es):
            tmp = convmf([m],e)[0]
            # some fudges to avoid -pi->pi etc. steps in grid
            if tmp > np.pi:
                tmp -= 2*np.pi
            if i == 0 and tmp == np.pi:
                tmp -= 2*np.pi
            f[i,j] = tmp

    return scipy.interpolate.RectBivariateSpline(Ms, es, f)


def convmf_fast(m_in, e_in, n=200):
    '''Convert mean to true anomaly with a lookup table.

    Parameters
    ----------
    m_in : float or ndarray
        Mean anomaly.
    e_in : float or ndarray
        Eccentricity.
    '''
    m = m_in % (2*np.pi)
    m[m>=np.pi] -= 2*np.pi
    convmf_interp = convmf_lookup(n=n)
    return convmf_interp.ev(m, e_in)


class Dens(object):
    '''Define some density functions.
        
    These are generally two or three-dimensional, in most cases with a
    Gaussian scale height that is either fixed (to 0.05 by default) or a
    variable. Since the scale height (z/r) is the parameter, rather than
    absolute height, the (z-integrated) surface density will increase 
    with radius unless explicitly accounted for, e.g. most Gaussian torus
    models are Gaussian in 3d space, not in surface density, but for the
    power-law models the index is for surface density.
    
    These models should peak at or near 1, to aid finding the integration
    box in `compute_rmax`.
    '''

    def __init__(self,model='gauss_3d',gaussian_scale_height=0.05,
                 box_half_height=0.05,
                 func=None, params=None, p_ranges=None):
        '''Get an object to do density.
        
        Parameters
        ----------
        model : str
            Name of model to use.
        gaussian_scale_height: float
            Scale height to use for fixed-height models.
        box_half_height: float
            Height to use for fixed-height models.
        func : function
            Density fuction to use.
        params : list of str
            Names of parameters in given func.
        p_ranges : list of pairs
            Allowed ranges of given parameters.
        '''
        self.gaussian_scale_height = gaussian_scale_height
        self.box_half_height = box_half_height
        self.select(model=model, func=func, params=params,
                    p_ranges=p_ranges)


    def select(self,model=None, func=None, params=None, p_ranges=None,
               list_models=False):
        '''Select a density model.
        
        Parameters
        ----------
        model : str
            Name of model to use, one of list returned by list_models.
        func : function
            Custom function to use, takes r, az (rad), el (rad), par.
        params : list
            List of parameter names associated with custom function.
        p_ranges : list of two-element lists
            List of ranges for each parameter.
        list_models : bool, optional
            Return list of available models.
        '''

        models = {
            'gauss_3d':{'func':self.gauss_3d,
                        'params':self.gauss_3d_params,
                        'p_ranges':self.gauss_3d_p_ranges},
            'gauss_surf_3d':{'func':self.gauss_surf_3d,
                        'params':self.gauss_surf_3d_params,
                        'p_ranges':self.gauss_surf_3d_p_ranges},
            'gauss_ecc_3d':{'func':self.gauss_ecc_3d,
                            'params':self.gauss_ecc_3d_params,
                            'p_ranges':self.gauss_ecc_3d_p_ranges},
            'gauss_ecc_2d':{'func':self.gauss_ecc_2d,
                            'params':self.gauss_ecc_2d_params,
                            'p_ranges':self.gauss_ecc_2d_p_ranges},
            'gauss_2d':{'func':self.gauss_2d,
                        'params':self.gauss_2d_params,
                        'p_ranges':self.gauss_2d_p_ranges},
            'gauss2_3d':{'func':self.gauss2_3d,
                        'params':self.gauss2_3d_params,
                        'p_ranges':self.gauss2_3d_p_ranges},
            'gauss_3d_test':{'func':self.gauss_3d_test,
                             'params':self.gauss_3d_test_params,
                             'p_ranges':self.gauss_3d_test_p_ranges},
            'power_2d':{'func':self.power_2d,
                        'params':self.power_2d_params,
                        'p_ranges':self.power_2d_p_ranges},
            'power_3d_ecc_rin':{'func':self.power_3d_ecc_rin,
                                'params':self.power_3d_ecc_rin_params,
                                'p_ranges':self.power_3d_ecc_rin_p_ranges},
            'power_2d_ecc_rin':{'func':self.power_2d_ecc_rin,
                                'params':self.power_2d_ecc_rin_params,
                                'p_ranges':self.power_2d_ecc_rin_p_ranges},
            'power_3d':{'func':self.power_3d,
                        'params':self.power_3d_params,
                        'p_ranges':self.power_3d_p_ranges},
            'power2_3d':{'func':self.power2_3d,
                         'params':self.power2_3d_params,
                         'p_ranges':self.power2_3d_p_ranges},
            'power2_top_3d':{'func':self.power2_top_3d,
                         'params':self.power2_top_3d_params,
                         'p_ranges':self.power2_top_3d_p_ranges},
            'box_2d':{'func':self.box_2d,
                        'params':self.box_2d_params,
                        'p_ranges':self.box_2d_p_ranges},
            'box_3d':{'func':self.box_3d,
                        'params':self.box_3d_params,
                        'p_ranges':self.box_3d_p_ranges},
            'peri_glow':{'func':self.peri_glow,
                        'params':self.peri_glow_params,
                        'p_ranges':self.peri_glow_p_ranges},
                  }
    
        if list_models:
            return list(models.keys())

        if func is None:
            self.model = model
            self.dens = models[model]['func']
            self.params = models[model]['params']
            self.p_ranges = models[model]['p_ranges']
        elif func is not None and params is not None:
            if model is None:
                self.model = 'custom'
            else:
                self.model = model
            self.dens = func
            self.params = params
            self.p_ranges = p_ranges
        else:
            raise ValueError('incorrect arguments')

    # set the allowed ranges for:
    # radius, width, height, power exponent, eccentricity
    rr = [0.,np.inf]
    dr = [0.001,np.inf]
    dh = [0.01,1.] # radians
    pr = [-np.inf,np.inf]
    er = [0.,0.5]

    # pericenter glow, placeholder for now
    peri_glow_params = ['$a_0$', '$\sigma_a$', '$e_f$', '$i_f$', '$e_p$',
                        '$\sigma_{e,p}$', '$\sigma_{i,p}$']
    peri_glow_p_ranges = [rr,dr,er,er,er,er,er]
    def peri_glow(self, r, az, el, p):
        '''Placeholder for pericenter glow.'''
        print('Pericenter glow model not available by this method.')
    
    # Gaussian torus and parameters
    gauss_3d_params = ['$r_0$','$\sigma_r$','$\sigma_h$']
    gauss_3d_p_ranges = [rr,dr,dh]
    def gauss_3d(self, r, az, el, p):
        '''Gaussian torus.'''
        return np.exp( -0.5*( (r-p[0])/p[1] )**2 ) * \
                    np.exp( -0.5*(el/p[2])**2 )

    # Gaussian torus, Gaussian surface density
    gauss_surf_3d_params = ['$r_0$','$\sigma_r$','$\sigma_h$']
    gauss_surf_3d_p_ranges = [rr,dr,dh]
    def gauss_surf_3d(self, r, az, el, p):
        '''Gaussian torus.'''
        return np.exp( -0.5*( (r-p[0])/p[1] )**2 ) * \
                    np.exp( -0.5*(el/p[2])**2 ) / r

    # Gaussian torus with fixed scale height and parameters
    gauss_2d_params = ['$r_0$','$\sigma_r$']
    gauss_2d_p_ranges = [rr,dr]
    def gauss_2d(self, r, az, el, p):
        '''Gaussian torus with fixed scale height.'''
        return self.gauss_3d(r,az,el,np.append(p,self.gaussian_scale_height))

    # Gaussian in/out torus and parameters
    gauss2_3d_params = ['$r_0$','$\sigma_{r,in}$',
                        '$\sigma_{r,out}$','$\sigma_h$']
    gauss2_3d_p_ranges = [rr,dr,dr,dh]
    def gauss2_3d(self,r,az,el,p):
        '''Gaussian torus, independent inner and outer sigma.'''
        dens = np.zeros(r.shape)
        ok = r < p[0]
        if np.any(ok):
            dens[ok] = np.exp( -0.5*( (r[ok]-p[0])/p[1] )**2 ) * \
                       np.exp( -0.5*(el[ok]/p[3])**2 )
        out = np.invert(ok)
        if np.any(out):
            dens[out] = np.exp( -( 0.5*(r[out]-p[0])/p[2] )**2 ) * \
                        np.exp( -0.5*(el[out]/p[3])**2 )
        return dens

    # Gaussian torus with wierd azimuthal dependence and parameters
    gauss_3d_test_params = ['$r_0$','$\sigma_r$','$\sigma_h$']
    gauss_3d_test_p_ranges = [rr,dr,dh]
    def gauss_3d_test(self,r,az,el,p):
        '''Gaussian torus with a test azimuthal dependence.'''
        return np.exp( -0.5*( (r-p[0])/p[1] )**2 ) * \
                    np.exp( -0.5*(el/p[2])**2 ) * \
                    (az+2*np.pi)%(2*np.pi)

    # narrow Gaussian eccentric ring
    gauss_ecc_3d_params = ['$r_0$','$e$','$\sigma_r$','$\sigma_h$']
    gauss_ecc_3d_p_ranges = [rr,[0,1],dr,dh]
    def gauss_ecc_3d(self,r,az,el,p):
        '''Gaussian eccentric torus, variable width.'''
        r_ecc = p[0] * ( 1 - p[1]**2 ) / ( 1 + p[1]*np.cos(az) )
        return np.exp( -0.5*((r-r_ecc)/p[2])**2 )/np.sqrt(2*np.pi)/p[2] * \
               np.exp( -0.5*( el/p[3] )**2 ) * \
               (1 - p[1]*np.cos(az))

    # narrow Gaussian eccentric ring
    gauss_ecc_2d_params = ['$r_0$','$e$','$\sigma_r$']
    gauss_ecc_2d_p_ranges = [rr,[0,1],dr]
    def gauss_ecc_2d(self,r,az,el,p):
        '''Gaussian eccentric torus, variable width.'''
        return self.gauss_ecc_3d(r,az,el,np.append(p,self.gaussian_scale_height))

    # Single power law torus and parameters
    power_3d_params = ['$r_{in}$','$r_{out}$','$\\alpha$','$\sigma_h$']
    power_3d_p_ranges = [rr,rr,pr,dh]
    def power_3d(self,r,az,el,p):
        '''Single power law surface density profile with Gaussian scale
        height.'''
        in_i = (r > p[0]) & (r < p[1])
        if isinstance(in_i,(bool,np.bool_)):
            return float(in_i) * np.exp( -0.5*(el/p[3])**2 )
        else:
            dens = np.zeros(r.shape)
            dens[in_i] = r[in_i]**(p[2]-1) * np.exp( -0.5*(el[in_i]/p[3])**2 )
            return dens

    # Single power law torus and parameters
    power_2d_params = ['$r_{in}$','$r_{out}$','$\\alpha$']
    power_2d_p_ranges = [rr,rr,pr]
    def power_2d(self,r,az,el,p):
        '''Single power law surface density profile with fixed Gaussian
        scale height.'''
        return self.power_3d(r,az,el,np.append(p,self.gaussian_scale_height))

    # Single power law torus with eccentric inner edge and parameters
    power_3d_ecc_rin_params = ['$r_{in}$','$r_{out}$','$\\alpha$',
                               '$e_{in}$','$\sigma_h$']
    power_3d_ecc_rin_p_ranges = [rr,rr,pr,er,dh]
    def power_3d_ecc_rin(self,r,az,el,p):
        '''Single power law surface density profile with Gaussian scale 
        height and eccentric inner edge.'''
        r_in = p[0] * ( 1 - p[3]**2 ) / ( 1 + p[3]*np.cos(az) )
        in_i = (r > r_in) & (r < p[1])
        if isinstance(in_i,(bool,np.bool_)):
            return float(in_i) * np.exp( -0.5*(el/p[4])**2 )
        else:
            dens = np.zeros(r.shape)
            dens[in_i] = r[in_i]**(p[2]-1) * np.exp( -0.5*(el[in_i]/p[4])**2 )
            return dens

    # Single power law torus with eccentric inner edge and parameters
    power_2d_ecc_rin_params = ['$r_{in}$','$r_{out}$','$\\alpha$',
                               '$e_{in}$']
    power_2d_ecc_rin_p_ranges = [rr,rr,pr,er]
    def power_2d_ecc_rin(self,r,az,el,p):
        '''Single power law radial profile with fixed Gaussian scale height
        and eccentric inner edge.'''
        return self.power_3d_ecc_rin(r,az,el,np.append(p,self.gaussian_scale_height))

    # Two-power law torus and parameters
    power2_3d_params = ['$r_0$','$p_{in}$','$p_{out}$','$\sigma_h$']
    power2_3d_p_ranges = [rr,pr,pr,dh]
    def power2_3d(self,r,az,el,p):
        '''Two-power law surface density profile with Gaussian scale
        height.'''
        return 1/np.sqrt( (r/p[0])**p[2] + (r/p[0])**(-p[1]) ) * \
                    np.exp( -0.5*(el/p[3])**2 ) / r

    # Two-power top-hat law torus and parameters
    power2_top_3d_params = ['$r_0$','$p_{in}$','$p_{out}$',
                           '$\delta_r$','$\sigma_h$']
    power2_top_3d_p_ranges = [rr,pr,pr,dr,dh]
    def power2_top_3d(self,r,az,el,p):
        '''Two-power law top hat surface density profile with Gaussian
        scale height.'''
        hw = p[3]/2.0
        w2 = p[3]**2
        return np.sqrt( (2+w2) / ( (r/(p[0]+hw))**(2*p[2]) + w2 +
                                   (r/(p[0]-hw))**(-2*p[1]) ) ) * \
                  np.exp( -0.5*(el/p[4])**2 ) / r

    # Box torus and parameters
    box_2d_params = ['$r_{in}$','$r_{out}$','$\\alpha$']
    box_2d_p_ranges = [rr,rr,pr]
    def box_2d(self,r,az,el,p):
        '''Box torus in 2d with fixed height.'''
        return self.box_3d(r,az,el,np.append(p,self.box_half_height))

    # Box torus and parameters
    box_3d_params = ['$r_{in}$','$r_{out}$','$\\alpha$','$\delta_h$']
    box_3d_p_ranges = [rr,rr,pr,dh]
    def box_3d(self,r,az,el,p):
        '''Box torus in 3d.'''
        in_i = (r > p[0]) & (r < p[1]) & \
                   (np.abs(el) <= p[3]/2.0)
        if isinstance(in_i,(bool,np.bool_)):
            return float(in_i)
        else:
            dens = np.zeros(r.shape)
            dens[in_i] = r[in_i]**(p[2]-1)
            return dens


class Emit(object):
    '''Define some emission functions.'''

    def __init__(self, model='blackbody'):
        '''Get an object to do emission properties.
        
        Parameters
        ----------
        model : str, optional
            Name of emission model to use.
        '''
        self.select(model)


    def select(self, model=None, func=None, params=None, p_ranges=None,
               list_models=False):
        '''Select an emission model.
        
        Parameters
        ----------
        model : str
            Name of model to use, one of list returned by list_models.
        func : function
            Custom function to use, takes r, par.
        params : list
            List of parameter names associated with custom function.
        p_ranges : list of two-element lists
            List of ranges for each parameter.
        list_models : bool, optional
            Return list of available models.
        '''

        models = {
            'rj_tail':{'func':self.rj_tail,
                       'params':self.rj_tail_params,
                       'p_ranges':self.rj_tail_p_ranges},
            'blackbody':{'func':self.blackbody,
                         'params':self.blackbody_params,
                         'p_ranges':self.blackbody_p_ranges},
            'constant':{'func':self.constant,
                        'params':self.constant_params,
                        'p_ranges':self.constant_p_ranges}
                  }

        if list_models:
            return list(models.keys())

        if func is None:
            self.model = model
            self.emit = models[model]['func']
            self.params = models[model]['params']
            self.p_ranges = models[model]['p_ranges']
        elif func is not None and params is not None:
            if model is None:
                self.model = 'custom'
            else:
                self.model = model
            self.emit = func
            self.params = params
            self.p_ranges = p_ranges
        else:
            raise ValueError('incorrect arguments')


    # blackbody, no knobs, p is a dummy
    rj_tail_params = []
    rj_tail_p_ranges = []
    def rj_tail(self, r, p, wavelength=None):
        '''Rayleigh-Jeans tail.'''
        return 1.0/r**0.5

    # blackbody
    blackbody_params = ['$r_{T_0}$','$T_0$']
    blackbody_p_ranges = []
    def blackbody(self, r, p, wavelength=None):
        '''Blackbody, wavelength in m.'''
        k1 = 3.9728949e19
        k2 = 14387.69
        temp = p[1] / np.sqrt(r/p[0])
        fact1 = k1/((wavelength*1e6)**3)
        fact2 = k2/(temp*wavelength*1e6)
        fnu = np.array(fact1/(np.exp(fact2)-1.0))
        return fnu
        
    # constant temp, no knobs, p is a dummy
    constant_params = []
    constant_p_ranges = []
    def constant(self, r, p, wavelength=None):
        '''Constant.'''
        return 1.0


class Image(object):
    '''Image generation.'''

    def __init__(self,image_size=None, arcsec_pix=None,
                 rmax_arcsec=None, rmax_off=(0,0),
                 model='los_image_axisym', emit_model='rj_tail',
                 dens_model='gauss_3d', dens_args={},
                 wavelength=None, pb_diameter=12.0, pb_fits=None,
                 automask=False,
                 star=False, z_fact=1, verbose=True):
        '''Get an object to make images.

        Parameters
        ----------
        image_size: length 2 tuple (nx, ny)
            Size of output image. Better to be even.
        arcsec_pix: float
            Pixel scale of output image.
        rmax_arcsec: float
            Maximum x,y,z extent of model, for speed.
        rmax_off : tuple, optional
            Offset for rmax.
        zfact : int, optional
            Factor to increase z resolution by.
        model: str
            Integration model to use; includes anomaly or not
        dens_model: str
            Density model to use. Takes some parameters
        dens_args : dict
            Dict of args to be passed to dens.
        emit_model: str
            Emission model to use. Takes no parameters.
        automask : bool
            Do automasking to speed up imaging when computing rmax.
        star : bool, optional
            Include a star at the image center.
        wavelength : float
            Wavelength of observations in m, used to create primary beam.
        pb_diameter : float, optional
            Dish diameter to calculate primary beam, in m.
        pb_fits : str
            Path to fits file of primary beam.
        '''

        if image_size[0] % 2 != 0 or image_size[1] % 2 != 0:
            print('WARNING: image size {} not even,'
                  'interpret offsets carefully!'.format(image_size))

        self.image_size = image_size
        self.automask = automask
        self.model = model
        self.emit_model = emit_model
        self.dens_model = dens_model
        self.arcsec_pix = arcsec_pix
        self.rad_pix = arcsec_pix / (3600*180/np.pi)
        self.wavelength = wavelength
        self.pb_diameter = pb_diameter
        self.z_fact = z_fact
        self.verbose = verbose

        # set fixed some things needed to make images
        self.nx, self.ny = self.image_size
        self.nx2 = self.nx // 2
        self.ny2 = self.ny // 2

        # set the image model
        self.select(model)
        
        # primary beam, first set radial function
        if pb_fits is None:
            self.primary_beam_func = self.analytic_primary_beam_func
        else:
            self.primary_beam_func = self.get_empirical_primary_beam_func(pb_fits)

        if self.wavelength is None and pb_fits is None:
            self.pb = None
            self.pb_galario = None
        elif self:
            self.set_primary_beam()

        # set r_max
        if rmax_arcsec is None:
            rmax = np.array([self.nx2,self.ny2,
                             np.max([self.nx2,self.ny2])])
            self.set_rmax(rmax)
        else:
            self.set_rmax(rmax_arcsec/arcsec_pix,
                          x0=rmax_off[0], y0=rmax_off[1])

        # set the density distribution function
        d = Dens(model=dens_model,**dens_args)
        self.Dens = d
        self.dens = d.dens
        self.dens_params = d.params
        self.dens_p_ranges = d.p_ranges
        self.n_dens_params = len(self.dens_params)

        # set the emission properties function
        e = Emit(model=emit_model)
        self.Emit = e
        self.emit = e.emit
        self.emit_params = e.params
        self.emit_p_ranges = e.p_ranges
        self.n_emit_params = len(self.emit_params)

        # paste the params together
        self.params = self.image_params + self.emit_params + self.dens_params
        self.p_ranges = self.image_p_ranges + self.emit_p_ranges + self.dens_p_ranges
        self.n_params = len(self.params)

        # decide if there is a star or not
        self.star = star
        if star:
            self.params += ['$F_{star}$']
            self.p_ranges += [[0.0, np.inf]]
            self.n_params = len(self.params)

        # say something about the model
        if self.verbose:
            print('model:{} with density:{} and emit:{}'.\
                    format(model,dens_model,emit_model))
            print('parameters are {}'.format(self.params))
            if rmax_arcsec is None:
                print('rmax not set, run compute_rmax before generating images (may not apply to all density models)')


    def select(self,model):
        '''Select the model we want to use.
        
        Parameters
        ----------
        model : str
            Name of model to use for image generation.
        '''

        models = {
            'los_image':{'fit_func':self.los_image_galario,
                         'img_func':self.los_image,
                         'cutout_func':self.los_image_cutout,
                         'cube':self.los_image_cube,
                         'rv_cube':self.rv_cube_,
                         'rv_cube_gal':self.rv_cube_galario_,
                         'params':self.los_image_params,
                         'p_ranges':self.los_image_p_ranges
                        },
            'los_image_axisym':{'fit_func':self.los_image_galario_axisym,
                                'img_func':self.los_image_axisym,
                                'cutout_func':self.los_image_cutout_axisym,
                                'cube':self.los_image_cube_axisym,
                                'rv_cube':self.rv_cube_axisym,
                                'rv_cube_gal':self.rv_cube_galario_axisym,
                                'params':self.los_image_axisym_params,
                                'p_ranges':self.los_image_axisym_p_ranges
                        }
                  }

        self.image = models[model]['img_func']
        self.image_galario = models[model]['fit_func']
        self.image_cutout = models[model]['cutout_func']
        self.cube = models[model]['cube']
        self.rv_cube = models[model]['rv_cube']
        self.rv_cube_galario = models[model]['rv_cube_gal']
        self.image_params = models[model]['params']
        self.image_p_ranges = models[model]['p_ranges']
        self.n_image_params = len(self.image_params)


    def compute_rmax(self, p, tol=1e-5, expand=10,
                     radial_only=False, zero_node=False,
                     automask=False):
        '''Figure out model extent to make image generation quicker.
        
        This routine may be setting the extent used for an entire mcmc
        run, so some wiggle room should be allowed for so that the model
        doesn't move outside the space derived here.
        
        .. todo: check r < nx2 here
        
        .. todo: allow z > image size

        Parameters
        ----------
        p : list or tuple
            Full list of model parameters.
        tol : float, optional
            Level at which image is considered zero. For radial_only
            this relies on the density fuction having a peak near 1,
            otherwise it is relative to the peak when summed along the
            relevant axis.
        expand : int, optional
            Number of pixels to pad the image by.
        radial_only : bool, optional
            Do just a radial calculation of the limits.
        zero_node : bool, optional
            Set ascending node to zero, for galario modelling.
        '''
    
        if zero_node:
            p_ = np.append(np.append(p[:2],0), p[3:])
        else:
            p_ = p

        p_start = self.n_image_params + self.n_emit_params

        # Work inwards, stopping when density becomes non-zero. Density
        # model is assumed to peak near 1 so absolute tolerance is used.
        def find_radial():
            for r in np.arange(np.max([self.nx2,self.ny2]),1,-1):
                for az in np.arange(0,360,30):
                    for el in np.arange(-90,90,15):
                        if self.dens(r*self.arcsec_pix,az,el,
                                     p_[p_start:]) > tol:
                            self.set_rmax(np.tile(r + expand,3),
                                          x0=p[0], y0=p[1])
                            if self.verbose:
                                print('radial r_max: {} pix at {},{}'.\
                                      format(r,p[0]/self.arcsec_pix,
                                             p[1]/self.arcsec_pix))
                            return None
    
        find_radial()

        # get centered cutout cube of disk and compute limits from that
        if not radial_only:

            cube = self.image_cutout(p_, cube=True)
            if np.sum(cube) == 0:
                raise ValueError('empty cube, check parameters')
        
            # find model extents, zarray may be higher resolution by
            # factor z_fact, so account for this here
            x_sum = np.sum(cube,axis=(0,2))
            xmax = np.max( np.abs( np.where( x_sum/np.max(x_sum) > tol )[0] \
                                   - len(x_sum)//2 ) )
            y_sum = np.sum(cube,axis=(1,2))
            ymax = np.max(  np.abs( np.where( y_sum/np.max(y_sum) > tol )[0] \
                                    - len(y_sum)//2 ) )
            z_sum = np.sum(cube,axis=(0,1))
            zmax = np.max( np.abs( np.where( z_sum/np.max(z_sum) > tol )[0] \
                                   - len(z_sum)//2 ) )
            if self.verbose:
                print('model x,y,z extent {}, {}, {}'.format(xmax,ymax,zmax))

            rmax = ( np.array([xmax, ymax, zmax]) + expand ).astype(int)
            self.set_rmax(rmax, x0=p[0], y0=p[1])

            if automask or self.automask:
                self.mask_setup(np.sum(cube, axis=2), self.arcsec_pix)
            else:
                self.mask = None


    def set_rmax(self, rmax, x0=0, y0=0):
        '''Set rmax and associated things.
        
        Calculate model limits in full image, and set up some arrays.
        
        Parameters
        ----------
        rmax : np.ndarray
            One or three integers giving the pixel extent of the model.
        x0 : int
            shift in arcsec (when models are offset in image).
        y0 : int
            shift in arcsec (when models are offset in image).
        '''

        # check input
        if not isinstance(rmax,np.ndarray):
            raise TypeError('please pass set_rmax an np.ndarray')
        if len(rmax) == 1:
            rmax = np.tile(rmax,3)

        # rmax is two-sided as cutout can be off center
        x0_pix = int(round(x0/self.arcsec_pix))
        y0_pix = int(round(y0/self.arcsec_pix))
        self.x0_pix = int(round(x0_pix))
        self.y0_pix = int(round(y0_pix))
        self.x0_arcsec = x0_pix * self.arcsec_pix
        self.y0_arcsec = y0_pix * self.arcsec_pix
        off = np.array([self.x0_pix, self.y0_pix, 0])
        self.rmax = np.array([off - rmax.astype(int),
                              off + rmax.astype(int)]).T

        # check image is not cropped
        if np.sum([self.rmax[0,0] < -self.nx2, self.rmax[0,1] > self.nx2,
                   self.rmax[1,0] < -self.ny2, self.rmax[1,1] > self.ny2]):
            raise ValueError('crop outside image, make larger or pad less')

        # crop slices for convenience, add one for Galario
        # cutout to allow for off-center image
        self.cc_gal = (slice(self.ny2-rmax[1]+1,self.ny2+rmax[1]),
                       slice(self.nx2-rmax[0]+1,self.nx2+rmax[0]))
        self.cc = (slice(self.ny2+self.rmax[1,0],self.ny2+self.rmax[1,1]),
                   slice(self.nx2+self.rmax[0,0],self.nx2+self.rmax[0,1]))

        # these arrays are centered on the crop, so not offset
        self.crop_size = np.diff(self.rmax).T[0]
        self.x = np.arange(self.crop_size[0]) - (self.crop_size[0]-1)/2.
        self.y = np.arange(self.crop_size[1]) - (self.crop_size[1]-1)/2.

        z_crop = int(self.crop_size[2] * self.z_fact)
        self.z = ( np.arange(z_crop) - (z_crop-1)/2. ) / self.z_fact
        self.zarray, self.yarray = np.meshgrid(self.z, self.y)
            
        # recompute primary beam, to include shift
        if self.wavelength is not None:
            self.set_primary_beam(x0=x0, y0=y0)


    def mask_setup(self, image, arcsec_pix, clip=1e-3):
        """Set up mask.

        Parameters
        ----------
        mask : 2d ndarray
            Image plane mask.
        arcsec_pix : float
            Pixel scale of image.
        clip : float
            Mask pixels below clip x peak.
        """
        from scipy.interpolate import RegularGridInterpolator

        if image.shape[0]*arcsec_pix < self.crop_size[0]*self.arcsec_pix or \
                image.shape[1]*arcsec_pix < self.crop_size[0]*self.arcsec_pix:
            print(f' Warning, image mask smaller than cutout')

        x = np.arange(image.shape[0]) - (image.shape[0]-1)/2.
        y = np.arange(image.shape[1]) - (image.shape[1]-1)/2.
        rg = RegularGridInterpolator((x*arcsec_pix, y*arcsec_pix), image, bounds_error=False, fill_value=0.0)

        x, y = np.meshgrid(self.x, self.y)
        xi = np.stack((y*self.arcsec_pix, x*self.arcsec_pix), axis=-1).reshape(-1, 2)
        mask = rg(np.array(xi)).reshape(self.crop_size[1], self.crop_size[0])

        mx = np.max(mask)
        mask[mask >= clip*mx] = 1
        mask[mask < clip*mx] = 0
        self.mask = np.array(mask, dtype=bool)


    def analytic_primary_beam_func(self, r):
        '''Analytic radial function for primary beam.
        
        Use Gaussian of FWHM 1.13 lambda/D, based on this:
        https://help.almascience.org/index.php?/Knowledgebase/Article/View/234
        
        Parameters
        ----------
        r : array of float
            Radius array, in radians.
        '''
        return np.exp(-0.5 * ( r / (1.13*self.wavelength/self.pb_diameter/2.35) )**2 )


    @staticmethod
    def get_empirical_primary_beam_func(file, verb=True):
        '''Return an interpolation object given a primary beam FITS.
        
        The object takes radius in radians.
        '''
        from astropy.io import fits
        from scipy.optimize import minimize
        from scipy.interpolate import interp1d

        # open fits and get pixel scale
        pb = fits.getdata(file)
        h = fits.open(file)
        pb = np.nan_to_num(pb.squeeze())
        nxy, _ = pb.squeeze().shape # assume square
        aspp = np.abs(h[0].header['CDELT1']*3600)

        def get_r(p, im):
            '''Get radius in radians.'''
            nx, ny = im.shape
            x = np.arange(nx) - (nx-1)/2. + p[0] / aspp
            y = np.arange(ny) - (ny-1)/2. + p[1] / aspp
            xx,yy = np.meshgrid(x,y)
            return np.sqrt(xx**2 + yy**2) * aspp/3600*np.pi/180

        def g2d(p, im):
            r = get_r(p, im)
            return np.exp(-0.5 * ( r / (p[2]*1e-4/2.35) )**2 )

        def chi2(p, im):
            return np.sum( ((im-g2d(p,im)))**2 )

        # fit Gaussian to find center
        p0 = [0, 0,  1.24226302]
        res = minimize(chi2, p0, args=(pb), method='Nelder-Mead',
                       options={'xatol':1e-5, 'fatol':1e-5, 'maxiter':1000, 'maxfev':1000})
        if verb:
            print('Initial:{}:{}'.format(p0, chi2(p0, pb)))
            print('Gaussian fit:{}'.format(res))

        # create interpolation object
        r_rad = get_r(res['x'], pb)
        r_int = np.append(0, r_rad.flatten())
        pb_int = np.append(1, pb.flatten())
        return interp1d(r_int, pb_int, fill_value=0.0, bounds_error=False)


    def primary_beam_image(self, x0=0, y0=0):
        '''Return an image of the primary beam.

        Parameters
        ----------
        x0 : float
            shift in arcsec (when models are offset in image).
        y0 : float
            shift in arcsec (when models are offset in image).
        '''
        x = np.arange(self.nx) - (self.nx-1)/2. - x0 / self.arcsec_pix
        y = np.arange(self.ny) - (self.ny-1)/2. - y0 / self.arcsec_pix
        xx,yy = np.meshgrid(x,y)
        r = np.sqrt(xx**2 + yy**2) * self.rad_pix
        return self.primary_beam_func(r)


    def set_primary_beam(self, x0=0, y0=0):
        '''Set images of the primary beam.
            
        This relies on the radial function having been set elsewhere.

        Parameters
        ----------
        x0 : float
            shift in arcsec (when models are offset in image).
        y0 : float
            shift in arcsec (when models are offset in image).
        '''
        
        # primary beam for normal images
        self.pb = self.primary_beam_image()
        
        # same again for galario images, the shift is negative since the
        # galario image is not shifted, and will be shifted by galario
        # (i.e. the primary beam is pre-shifted to attenuate the off-
        # center image)
        self.pb_galario = self.primary_beam_image(-x0, -y0)
    

    def los_image_cutout_(self, p, cube=False):
        '''Return a cutout image of a disk.

        This is ultimately based on the zodipic code.
        
        Parameters
        ----------
        p : list
            List of parameters, x0, y0, pos, anom, inc, tot, + emit/dens.
            Distances are in arcsec and angles in degrees.
        cube : bool, optional
            Return cutout image cube (y,z,x) for use elsewhere.
        '''
        
        x0, y0, pos, anom, inc, tot = p[:6]

        yarray = self.yarray - y0 / self.arcsec_pix

        # zyz rotation, a.c.w so use -ve angles for cube -> model
        c0 = np.cos(np.deg2rad(-pos))
        s0 = np.sin(np.deg2rad(-pos))
        c1 = np.cos(np.deg2rad(-inc))
        s1 = np.sin(np.deg2rad(-inc))
        c2 = np.cos(np.deg2rad(-anom)-np.pi/2) # to get from N to x
        s2 = np.sin(np.deg2rad(-anom)-np.pi/2)
        c0c1c2 = c0 * c1 * c2
        c0c1s2 = c0 * c1 * s2
        c0s1 = c0 * s1
        s0s2 = s0 * s2
        s0c1 = s0 * c1
        s0c2 = s0 * c2
        c0c1c2_s0s2 = c0c1c2 - s0s2
        c0c1s2_s0c2 = c0c1s2 + s0c2

        # x-independent parts of the coordinate transformation
        trans1 = -(s0c1*c2 + c0*s2)*yarray + s1*c2*self.zarray
        trans2 = (-s0c1*s2 + c0*c2)*yarray + s1*s2*self.zarray
        trans3 = s0*s1*yarray + c1*self.zarray

        # cube method, ~50% slower than by layers below
        if cube:
            x = self.x - x0 / self.arcsec_pix
            x3 = c0c1c2_s0s2*x + trans1[:,:,None]
            y3 = c0c1s2_s0c2*x + trans2[:,:,None]
            z3 = -c0s1*x + trans3[:,:,None]

            # get the spherical polars
            rxy2 = x3**2 + y3**2
            rxy = np.sqrt(rxy2)
            r = np.sqrt(rxy2 + z3**2) * self.arcsec_pix
            if 'axisym' in self.model:
                az = 0.
            else:
                az = np.arctan2(y3,x3)
            el = np.arctan2(z3,rxy)

            # the density, dimensions are y, z, x -> y, x, z
            cube = self.emit(r,p[slice(6,6+self.n_emit_params)],
                             wavelength=self.wavelength) * \
                   self.dens(r,az,el,p[6+self.n_emit_params:])
            return tot * np.rollaxis(cube, 2, 1) / np.sum(cube)
    
        # go through slice by slice and get the flux along an x column
        # attempts to speed this up using multiprocess.Pool failed
        else:

            image = np.zeros((self.crop_size[1],self.crop_size[0]))

            for i in np.arange(self.crop_size[0]):

                x = self.x[i] - x0 / self.arcsec_pix

                # x,y,z locations in original model coords
                if self.mask is not None:
                    ind = self.mask[:, i]
                    x3 = c0c1c2_s0s2*x + trans1[ind]
                    y3 = c0c1s2_s0c2*x + trans2[ind]
                    z3 = -c0s1*x + trans3[ind]
                else:
                    x3 = c0c1c2_s0s2*x + trans1
                    y3 = c0c1s2_s0c2*x + trans2
                    z3 = -c0s1*x + trans3

                # get the spherical polars
                rxy2 = x3**2 + y3**2
                rxy = np.sqrt(rxy2)
                r = np.sqrt(rxy2 + z3**2) * self.arcsec_pix
                # these two lines are as expensive as dens below
                if 'axisym' in self.model:
                    az = 0.
                else:
                    az = np.arctan2(y3,x3)
                el = np.arctan2(z3,rxy)

                # the density in this y,z layer
                layer = self.emit(r,p[slice(6,6+self.n_emit_params)],
                                  wavelength=self.wavelength) * \
                        self.dens(r,az,el,
                                  p[slice(6+self.n_emit_params,
                                          6+self.n_emit_params+self.n_dens_params)]
                                  )

                # put this in the image
                if self.mask is not None:
                    image[ind, i] = np.sum(layer, axis=1)
                else:
                    image[:, i] = np.sum(layer, axis=1)

        image = tot * image / np.sum(image)
        
        # star, Gaussian 2pixel FWHM
        if self.star:
            if p[6+self.n_emit_params+self.n_dens_params] > 0.0:
                x, y = np.meshgrid(self.x - x0 / self.arcsec_pix,
                                   self.y - y0 / self.arcsec_pix)
                rxy2 = x**2 + y**2
                sigma = 2. / 2.35482
                image += p[6+self.n_emit_params+self.n_dens_params] * \
                         np.exp(-0.5*rxy2/sigma**2) / (2*np.pi*sigma**2)

        return image
                      

    def los_image_cutout(self, p, cube=False):
        '''Version of los_image_cutout_.
            
        All calls come through here so this is the only routine where
        the cutout offset is subtracted.
        '''
        return self.los_image_cutout_(np.append([p[0]-self.x0_arcsec,
                                                p[1]-self.y0_arcsec],
                                               p[2:]), cube=cube)

    def los_image_cutout_axisym(self, p, cube=False):
        return self.los_image_cutout(np.append(p[:3],np.append(0.0,p[3:])),
                                               cube=cube)

    los_image_params = ['$x_0$','$y_0$','$\Omega$','$\omega$','$i$','$F$']
    los_image_p_ranges = [[-np.inf,np.inf], [-np.inf,np.inf], [-270,270],
                          [-270,270], [0.,120], [0.,np.inf]]
    def los_image(self, p):
        '''Version of los_image, full parameters'''
        img = self.los_image_cutout(p)
        image = np.zeros((self.ny, self.nx))
        image[self.ny2+self.rmax[1,0]:self.ny2+self.rmax[1,1],
              self.nx2+self.rmax[0,0]:self.nx2+self.rmax[0,1]] = img
        return image

    los_image_axisym_params = ['$x_0$','$y_0$','$\Omega$','$i$','$F$']
    los_image_axisym_p_ranges = [[-np.inf,np.inf], [-np.inf,np.inf],
                                 [-270,270], [0.,120], [0.,np.inf]]
    def los_image_axisym(self, p):
        '''Version of los_image, no anomaly dependence in dens.'''
        return self.los_image(np.append(p[:3],np.append(0.0,p[3:])))

    def los_image_galario(self, p, cutout=False):
        '''Version of los_image for galario, no x/y offset, position angle.
        
        Galario expects the image center in the center of the pixel to
        the upper right of the actual center (if 0,0 is the lower left).
        
        Assume here that galario will be called with origin='lower', so
        that images do not need to be flipped (having 0,0 in the lower
        left corner.

        Parameters
        ----------
        p : list
            List of parameters.
        cutout : bool
            Return cutout rather than full image.
        '''
        img = self.los_image_cutout_(np.append([self.arcsec_pix/2.,
                                                self.arcsec_pix/2.,0.0],
                                               p))
        if cutout:
            return img
        image = np.zeros((self.ny, self.nx))
        dx = np.diff(self.rmax[0])[0]
        dy = np.diff(self.rmax[1])[0]
        image[self.ny2-dy//2:self.ny2+dy//2,
              self.nx2-dx//2:self.nx2+dx//2] = img
        return image

    def los_image_galario_axisym(self, p, cutout=False):
        '''Version of los_image for galario, no x/y offset, postion
        angle, or anomaly dependence.
        '''
        return self.los_image_galario(np.append([0.0],p), cutout=cutout)

    def los_image_cube(self, p):
        '''Version of los_image to return the cube.'''
        img = self.los_image_cutout(p, cube=True)
        c = np.zeros((self.ny, self.nx, img.shape[2]))
        c[self.ny2+self.rmax[1,0]:self.ny2+self.rmax[1,1],
          self.nx2+self.rmax[0,0]:self.nx2+self.rmax[0,1], :] = img
        return c

    def los_image_cube_axisym(self, p):
        '''Version of los_image to return the cube.'''
        return self.los_image_cube(np.append(p[:3],np.append(0.0,p[3:])))


    def rv_cube_cutout_(self, p, rv_min, dv, n_chan, mstar,
                        distance, v_sys=0.0, vff=0.0):
        '''Return a cutout velocity cube, units of km/s.
            
        Parameters
        ----------
        p : list
            List of parameters, as for los_image.
        rv_min : float
            Minimum radial velocity in km/s (lower edge of first bin).
        dv : float
            Width of velocity channels.
        n_chan : int
            Number of velocity channels.
        mstar : float
            Mass of star in Solar masses.
        distance : float
            Distance to star in parsecs.
        v_sys : float, optional
            Systemic velocity, in km/s.
        vff : float, optional
            Add "free-fall" (+ve -> star) component.
        '''

        x0, y0, pos, anom, inc, tot = p[:6]

        au = 1.496e11
        sc = self.arcsec_pix * distance * au

        # get the density cube
        c = self.los_image_cutout_(p, cube=True)

        # cube distances
        x, y, z = np.meshgrid(self.x-x0/self.arcsec_pix,
                              self.y-y0/self.arcsec_pix, self.z)
        r = np.sqrt(x*x + y*y + z*z)

        # get distance along major axis as y
        t = np.array([[np.cos(np.deg2rad(pos)),-np.sin(np.deg2rad(pos))],
                      [np.sin(np.deg2rad(pos)), np.cos(np.deg2rad(pos))]])
        XY = np.vstack( (x.reshape(-1), y.reshape(-1)) )
        xy = np.matmul(t.T, XY)
        x, y = xy.reshape( (2,) + x.shape )

        # velocities in each pixel for this inclination
        vr = cube.v_rad(y * sc, r * sc, inc, mstar) + v_sys

        # "free-fall" (+ve is towards star) component
        if vff != 0.0:
            phi = np.arcsin(z/r) # angle from sky plane to pixel
            vr += -vff * np.sin(phi)

            # as fraction of V_Kep (sqrt(2) would be free fall from inf)
#            g, msun = 6.67408e-11, 1.9891e30
#            vcirc = np.sqrt(g * msun * mstar / (r*sc)**3) / 1e3
#            v_ff = - vff * vcirc * np.sin(phi)
#            vr += v_ff

        # the velocity cube
        edges = rv_min + np.arange(n_chan+1) * dv
        h = np.digitize(vr, bins=edges)

        rvc = np.zeros((c.shape[0], c.shape[1], n_chan))
        for i in range(n_chan):
            ok = h == i+1
            if np.any(ok):
                mask = np.array(ok, dtype=int)
                rvc[:,:,i] = np.sum(c*mask, axis=2)

        return rvc


    def rv_cube_cutout(self, p, rv_min, dv, n_chan, mstar,
                       distance, v_sys=0.0, vff=0.0):
        '''Version of rv_cube_cutout_.
            
        All calls come through here so this is the only routine where
        the cutout offset is subtracted.
        '''
        return self.rv_cube_cutout_(np.append([p[0]-self.x0_arcsec,
                                               p[1]-self.y0_arcsec], p[2:]),
                                    rv_min, dv, n_chan, mstar, distance,
                                    v_sys, vff)

    def rv_cube_(self, p, rv_min, dv, n_chan, mstar,
                distance, v_sys=0.0, vff=0.0):
        '''Version of rv_cube, full parameters.'''
        rvc = self.rv_cube_cutout(p, rv_min, dv, n_chan, mstar,
                                  distance, v_sys, vff)
        # put this in the image
        c_out = np.zeros((self.ny, self.nx, n_chan))
        c_out[self.ny2+self.rmax[1,0]:self.ny2+self.rmax[1,1],
              self.nx2+self.rmax[0,0]:self.nx2+self.rmax[0,1], :] = rvc
        return c_out

    def rv_cube_axisym(self, p, rv_min, dv, n_chan, mstar,
                       distance, v_sys=0.0, vff=0.0):
        '''Version of rv_cube for axisymmetric disks.'''
        return self.rv_cube_(np.append(p[:3],np.append(0.0,p[3:])),
                             rv_min, dv, n_chan, mstar, distance,
                             v_sys, vff)

    def rv_cube_galario_(self, p, rv_min, dv, n_chan,
                         mstar, distance, v_sys=0.0, vff=0.0):
        '''Version of los_image_cube for galario, no x/y offset,
        postion angle.

        Galario expects the image center in the center of the pixel to
        the upper right of the actual center, so the center is half a
        pixel away from the actual image center.
        '''

        rvc = self.rv_cube_cutout_(np.append([self.arcsec_pix/2.,
                                              self.arcsec_pix/2.,
                                              0.0],p), rv_min, dv,
                                   n_chan, mstar, distance, v_sys, vff)
        c = np.zeros((self.ny, self.nx, n_chan))
        dx = np.diff(self.rmax[0])[0]
        dy = np.diff(self.rmax[1])[0]
        c[self.ny2-dy//2:self.ny2+dy//2,
          self.nx2-dx//2:self.nx2+dx//2, :] = rvc
        return c

    def rv_cube_galario_axisym(self, p, rv_min, dv, n_chan,
                               mstar, distance, v_sys=0.0, vff=0.0):
        '''Version of los_image_cube for galario, no x/y offset, postion
        angle, or anomaly dependence.
        '''
        return self.rv_cube_galario_(np.append([0.0],p), rv_min, dv,
                                     n_chan, mstar, distance, v_sys, vff)


def eccentric_ring_positions(a0, da, e_f0, i_f0, e_p0,
                             sigma_ep=0, sigma_ip=0,
                             omega_f0=0, node=0.0, inc=0.0, n=100000,
                             return_e=False, da_gauss=True):
    '''Return positions of particles in an eccentric ring model.
    
    Parameters
    ----------
    a0 : float
        Semi-major axis of ring particles.
    da : float
        Gaussian dispersion of ring width.
    e_f0 : float
        Forced eccentricity magnitude.
    i_f0 : float
        Forced inclination magnitude.
    e_p0 : float
        Proper eccentricity, distance from forced eccentricity.
    sigma_ep : float
        Gaussian dispersion of proper eccentricity.
    sigma_ip : float
        Gaussian dispersion of proper inclination.
    omega_f0 : float
        Forced pericenter in radians.
    node : float
        Position angle of disk, in radians.
    inc : float
        Sky inclination of disk plane, in radians.
    n : int
        Number of particles.
    return_e : bool
        Return eccentricity vectors instead of positions.
    da_gauss : bool
        da is Gaussian sigma if True, uniform a0-da/2..a0+da/2 if False.
    '''

    # complex forced eccentricity vector
    e_f = e_f0 * np.exp(1j*omega_f0)

    # semi-major axes
    if da_gauss:
        a = np.random.normal(loc=a0, scale=da, size=n)
    else:
        a = np.random.uniform(a0-da/2, a0+da/2, size=n)

    # proper eccentricity is distance from forced towards origin
    # (i.e. e_p=e_f means original e_p was at origin, not at forced e)
    e_p = np.abs(np.random.normal(scale=sigma_ep, size=n))
    omega_p = 2*np.pi*np.random.uniform(size=n)
    e_p_vec = e_p*np.exp(1j*omega_p)            # at zero
    e_p_vec += (e_f0 - e_p0)*np.exp(1j*omega_f0) # to actual location

    # pericenter angle distribution around forced eccentricity
    omega_ef = 2*np.pi*np.random.uniform(size=n)

    # mean anomaly
    M = 2*np.pi*np.random.uniform(size=n)

    # inclination, fixed around forced i for random node
    i = np.random.normal(loc=i_f0, scale=sigma_ip, size=n)

    # node, distributed randomly at ~fixed inclination
    # as expected for secular nodal precession
    Omega = 2*np.pi*np.random.uniform(size=n)

    # full complex eccentricity distribution
    e_vec = e_f + np.abs(e_p_vec - e_f) * np.exp(1j*omega_ef)
    e = np.abs(e_vec)
    omega_0 = np.angle(e_vec)

    if return_e:
        return e_f, e_p_vec, e_vec

    # true anomaly
    f = convmf_fast(M, e)

    # orbital locations, theta is from forced peri to particle
    r = a * (1-e**2)/(1+e*np.cos(f))
    theta = f + omega_0 - Omega

    # cartesians in 'model' coordinates (but omega_f included)
    x = r * (np.cos(Omega)*np.cos(theta) - np.sin(Omega)*np.sin(theta)*np.cos(i))
    y = r * (np.sin(Omega)*np.cos(theta) + np.cos(Omega)*np.sin(theta)*np.cos(i))
    z = r * np.sin(i) * np.sin(theta)

    # incline and rotate to observed position
    y_tmp = y * np.cos(inc)
    x_rot = x * np.cos(node+np.pi/2) - y_tmp * np.sin(node+np.pi/2)
    y_rot = x * np.sin(node+np.pi/2) + y_tmp * np.cos(node+np.pi/2)

    return x, y, x_rot, y_rot, z


def eccentric_ring_image(p, nxy, dxy_arcsec, n=100000,
                         star_fwhm=2, return_e=False, da_gauss=True):
    '''Return an image of the particle-based eccentric ring model.
    
    Assumes use of galario where zero is center of pixel above and right
    of actual center, when using origin='lower'.

    Parameters
    ----------
    p : list
        Parameter list.
    nxy : int
        Number of pixels across (square) image.
    dxy_arcsec : float
        Arcseconds per pixel.
    n : int, optional
        Number of particles used to create image.
    star_fwhm : float, optional
        FWHM of star (make small for it to be a single pixel).
    return_e : bool
        Return eccentricity vectors instead of an image.
    da_gauss : bool
        da is Gaussian sigma if True, uniform a0-da/2..a0+da/2 if False.
    '''

    # get the particles
    out = eccentric_ring_positions(p[6], p[7], p[8], p[9], p[10],
                                   sigma_ep=p[11], sigma_ip=p[12],
                                   omega_f0=np.deg2rad(p[3]),
                                   node=np.deg2rad(p[2]),
                                   inc=np.deg2rad(p[4]),
                                   n=n, return_e=return_e,
                                   da_gauss=da_gauss)
                                    
    if return_e:
        return out
    else:
        x0, y0, x, y, z0 = out

    # for weighting by temperature, use model coords
    r = np.sqrt(x0*x0 + y0*y0 + z0*z0)

    # flux normalised image
    x_arr = np.array([-(nxy/2+0.5)*dxy_arcsec, (nxy/2-0.5)*dxy_arcsec])
    x0 = x_arr - p[0]
    y0 = x_arr - p[1]
    h, _, _ = np.histogram2d(y, x, bins=nxy, range=[y0, x0],
                             weights=1/np.sqrt(r), density=True)
    h = p[5] * h / np.sum(h)

    # star if desired
    if len(p) == 14:
        xarr = np.arange(nxy)-nxy/2
        y, x = np.meshgrid(xarr - p[0]/dxy_arcsec, xarr - p[1]/dxy_arcsec)
        rxy2 = x**2 + y**2
        sigma = star_fwhm / 2.35482
        star = np.exp(-0.5*rxy2/sigma**2)
        h += star * p[13] / np.sum(star)

    return h
