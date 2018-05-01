from __future__ import print_function

import numpy as np

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

class Dens(object):
    '''Define some density functions.
    
    Should peak at or near 1, to aid finding integration box.
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
            'gauss_ecc_3d':{'func':self.gauss_ecc_3d,
                            'params':self.gauss_ecc_3d_params,
                            'p_ranges':self.gauss_ecc_3d_p_ranges},
            'gauss_2d':{'func':self.gauss_2d,
                        'params':self.gauss_2d_params,
                        'p_ranges':self.gauss_2d_p_ranges},
            'gauss_2d_opthk':{'func':self.gauss_2d_opthk,
                              'params':self.gauss_2d_opthk_params,
                              'p_ranges':self.gauss_2d_opthk_p_ranges},
            'gauss2_3d':{'func':self.gauss2_3d,
                        'params':self.gauss2_3d_params,
                        'p_ranges':self.gauss2_3d_p_ranges},
            'gauss_3d_test':{'func':self.gauss_3d_test,
                             'params':self.gauss_3d_test_params,
                             'p_ranges':self.gauss_3d_test_p_ranges},
            'power_3d':{'func':self.power_3d,
                        'params':self.power_3d_params,
                        'p_ranges':self.power_3d_p_ranges},
            'power_top_3d':{'func':self.power_top_3d,
                        'params':self.power_top_3d_params,
                        'p_ranges':self.power_top_3d_p_ranges},
            'box_2d':{'func':self.box_2d,
                        'params':self.box_2d_params,
                        'p_ranges':self.box_2d_p_ranges},
            'box_3d':{'func':self.box_3d,
                        'params':self.box_3d_params,
                        'p_ranges':self.box_3d_p_ranges},
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

    # set the allowed ranges for radius, width, height, power exponent
    rr = [0.,500.]
    dr = [0.001,10.]
    dh = [0.01,1.] # radians
    pr = [1.,50.]

    # Gaussian torus and parameters
    gauss_3d_params = ['$r_0$','$\sigma_r$','$\sigma_h$']
    gauss_3d_p_ranges = [rr,dr,dh]
    def gauss_3d(self, r, az, el, p):
        '''Gaussian torus.'''
        return np.exp( -0.5*( (r-p[0])/p[1] )**2 ) * \
                    np.exp( -0.5*(el/p[2])**2 )

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

    # Gaussian eccentric ring
    gauss_ecc_3d_params = ['$r_0$','$e$','$\sigma_{peri}$',
                           '$\sigma_{apo}/\sigma_{peri}$','$\sigma_h$']
    gauss_ecc_3d_p_ranges = [rr,[0,1],dr,[1,10],dh]
    def gauss_ecc_3d(self,r,az,el,p):
        '''Gaussian eccentric torus, variable width.'''
        r_ecc = p[0] * ( 1 - p[1]**2 ) / ( 1 + p[1]*np.cos(az) )
        w_ecc = p[2] + (p[3]-1)*p[2]*np.sin(az/2.)**2
        return np.exp( -0.5*((r-r_ecc)/w_ecc)**2 )/np.sqrt(2*np.pi)/w_ecc * \
               np.exp( -0.5*( el/p[4] )**2 ) * \
               (1 - p[1]*np.cos(az))

    # Power law torus and parameters
    power_3d_params = ['$r_0$','$p_{in}$','$p_{out}$','$\sigma_h$']
    power_3d_p_ranges = [rr,pr,pr,dh]
    def power_3d(self,r,az,el,p):
        '''Power law radial profile with Gaussian scale height.'''
        return 1/np.sqrt( (r/p[0])**p[2] + (r/p[0])**(-p[1]) ) * \
                    np.exp( -0.5*(el/p[3])**2 )

    # Power top-hat law torus and parameters
    power_top_3d_params = ['$r_0$','$p_{in}$','$p_{out}$',
                           '$\delta_r$','$\sigma_h$']
    power_top_3d_p_ranges = [rr,pr,pr,dr,dh]
    def power_top_3d(self,r,az,el,p):
        '''Power law top hat radial profile with Gaussian scale height.'''
        hw = p[3]/2.0
        w2 = p[3]**2
        return np.sqrt( (2+w2) / ( (r/(p[0]+hw))**(2*p[2]) + w2 +
                                   (r/(p[0]-hw))**(-2*p[1]) ) ) * \
                  np.exp( -0.5*(el/p[4])**2 )

    # Box torus and parameters
    box_2d_params = ['$r_0$','$\delta_r$']
    box_2d_p_ranges = [rr,dr]
    def box_2d(self,r,az,el,p):
        '''Box torus in 2d. assume r,az,el are vectors.'''
        return self.box_3d(r,az,el,np.append(p,self.box_half_height))

    # Box torus and parameters
    box_3d_params = ['$r_0$','$\delta_r$','$\delta_h$']
    box_3d_p_ranges = [rr,dr,dh]
    def box_3d(self,r,az,el,p):
        '''Box torus in 3d. assume r,az,el are vectors.'''
        in_i = (r > p[0]-p[1]/2.) & (r < p[0]+p[1]/2.) & \
                   (np.abs(el) <= p[2]/2)
        if isinstance(in_i,(bool,np.bool_)):
            return float(in_i)
        else:
            dens = np.zeros(r.shape)
            dens[in_i] = 1.0
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
    blackbody_params = []
    blackbody_p_ranges = []
    def blackbody(self, r, p):
        '''Blackbody.'''
        return 1.0/r**0.5

    # constant temp, no knobs, p is a dummy
    constant_params = []
    constant_p_ranges = []
    def constant(self, r, p):
        '''Constant.'''
        return 1.0


class Image(object):
    '''Image generation.'''

    def __init__(self,image_size=None, arcsec_pix=None,
                 rmax_arcsec=None, rmax_off=(0,0),
                 model='los_image_axisym', emit_model='blackbody',
                 dens_model='gauss_3d', dens_args={},
                 wavelength=None, pb_diameter=12.0,
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
        star : bool, optional
            Include a star at the image center.
        wavelength : float
            Wavelength of observations in m, used to create primary beam.
        pb_diameter : float, optional
            Dish diameter to calculate primary beam, in m.
        '''

        if image_size[0] % 2 != 0 or image_size[1] % 2 != 0:
            print('WARNING: image size {} not even,'
                  'interpret offsets carefully!'.format(image_size))

        self.image_size = image_size
        self.model = model
        self.emit_model = emit_model
        self.dens_model = dens_model
        self.arcsec_pix = arcsec_pix
        self.rad_pix = arcsec_pix / (3600*180/np.pi)
        self.wavelength = wavelength
        self.z_fact = z_fact

        # set fixed some things needed to make images
        self.nx, self.ny = self.image_size
        self.nx2 = self.nx // 2
        self.ny2 = self.ny // 2
        if rmax_arcsec is None:
            rmax = np.array([self.nx2,self.ny2,
                             np.max([self.nx2,self.ny2])])
            self.set_rmax(rmax)
        else:
            self.set_rmax(np.tile(rmax_arcsec/arcsec_pix,3),
                          x0=rmax_off[0], y0=rmax_off[1])

        # set the image model
        self.select(model)
        
        # generate the primary beam
        if self.wavelength is None:
            self.pb = None
            self.pb_galario = None
        else:
            self.set_primary_beam(self.wavelength)

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
        if verbose:
            print('model:{} with density:{} and emit:{}'.\
                    format(model,dens_model,emit_model))
            print('parameters are {}'.format(self.params))
            if rmax_arcsec is None:
                print('rmax not set, run compute_rmax before generating images')


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
                     radial_only=False, zero_node=False):
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
    
        # Work inwards, stopping when density becomes non-zero. Density
        # model is assumed to peak near 1 so absolute tolerance is used.
        def find_radial():
            for r in np.arange(np.max([self.nx2,self.ny2]),1,-1):
                for az in np.arange(0,360,30):
                    for el in np.arange(-90,90,15):
                        if self.dens(r*self.arcsec_pix,az,el,
                                     p_[self.n_image_params:]) > tol:
                            self.set_rmax(np.tile(r + expand,3),
                                          x0=p[0], y0=p[1])
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
            print('model x,y,z extent {}, {}, {}'.format(xmax,ymax,zmax))

            rmax = ( np.array([xmax, ymax, zmax]) + expand ).astype(int)
            self.set_rmax(rmax, x0=p[0], y0=p[1])


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

        self.cc_gal = (slice(self.ny2-rmax[1],self.ny2+rmax[1]),
                       slice(self.nx2-rmax[0],self.nx2+rmax[0]))
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
            self.set_primary_beam(self.wavelength, x0=x0, y0=y0)


    def set_primary_beam(self, wavelength, diameter=12.0, x0=0, y0=0):
        '''Create an image of the primary beam.
            
        Use Gaussian of FWHM 1.13 lambda/D, based on this:
        https://help.almascience.org/index.php?/Knowledgebase/Article/View/234
        
        Parameters
        ----------
        wavelength : float
            Wavelength of observation in m.
        diameter : float, optional
            Single dish diameter, default is 12m (i.e. ALMA).
        x0 : int
            shift in arcsec (when models are offset in image).
        y0 : int
            shift in arcsec (when models are offset in image).
        '''
        
        # primary beam for normal images
        x = np.arange(self.nx) - (self.nx-1)/2.
        y = np.arange(self.ny) - (self.ny-1)/2.
        xx,yy = np.meshgrid(x,y)
        r = np.sqrt(xx**2 + yy**2) * self.rad_pix
        
        self.pb = np.exp(-0.5 * ( r / (1.13*wavelength/diameter/2.35) )**2 )
        
        # same again for galario images
        x = np.arange(self.nx) - (self.nx-1)/2. + x0 / self.arcsec_pix
        y = np.arange(self.ny) - (self.ny-1)/2. + y0 / self.arcsec_pix
        xx,yy = np.meshgrid(x,y)
        r = np.sqrt(xx**2 + yy**2) * self.rad_pix
        
        self.pb_galario = np.exp(-0.5 * ( r / (1.13*wavelength/diameter/2.35) )**2 )
    

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
            az = np.arctan2(y3,x3)
            el = np.arctan2(z3,rxy)

            # the density, dimensions are y, z, x -> y, x, z
            cube = self.emit(r,p[slice(6,6+self.n_emit_params)]) * \
                   self.dens(r,az,el,p[6+self.n_emit_params:])
            return tot * np.rollaxis(cube, 2, 1) / np.sum(cube)
    
        # go through slice by slice and get the flux along an x column
        # attempts to speed this up using multiprocess.Pool failed
        else:

            image = np.zeros((self.crop_size[1],self.crop_size[0]))

            for i in np.arange(self.crop_size[0]):

                x = self.x[i] - x0 / self.arcsec_pix

                # x,y,z locations in original model coords
                x3 = c0c1c2_s0s2*x + trans1
                y3 = c0c1s2_s0c2*x + trans2
                z3 = -c0s1*x + trans3

                # get the spherical polars
                rxy2 = x3**2 + y3**2
                rxy = np.sqrt(rxy2)
                r = np.sqrt(rxy2 + z3**2) * self.arcsec_pix
                # these two lines are as expensive as dens below
                az = np.arctan2(y3,x3)
                el = np.arctan2(z3,rxy)

                # the density in this y,z layer
                layer = self.emit(r,p[slice(6,6+self.n_emit_params)]) * \
                        self.dens(r,az,el,
                                  p[slice(6+self.n_emit_params,
                                          6+self.n_emit_params+self.n_dens_params)]
                                  )

                # put this in the image
                image[:,i] = np.sum(layer,axis=1)

        image = tot * image / np.sum(image)
        
        # star, Gaussian 2pixel FWHM
        if self.star:
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

    los_image_params = ['$x_0$','$y_0$','$\Omega$','$f$','$i$','$F$']
    los_image_p_ranges = [[-1,1], [-1,1], [-270,270],
                          [-270,270], [0.,120], [0.,np.inf]]
    def los_image(self, p):
        '''Version of los_image, full parameters'''
        img = self.los_image_cutout(p)
        image = np.zeros((self.ny, self.nx))
        image[self.ny2+self.rmax[1,0]:self.ny2+self.rmax[1,1],
              self.nx2+self.rmax[0,0]:self.nx2+self.rmax[0,1]] = img
        return image

    los_image_axisym_params = ['$x_0$','$y_0$','$\Omega$','$i$','$F$']
    los_image_axisym_p_ranges = [[-1,1], [-1,1],
                                 [-270,270], [0.,120], [0.,np.inf]]
    def los_image_axisym(self, p):
        '''Version of los_image, no anomaly dependence in dens.'''
        return self.los_image(np.append(p[:3],np.append(0.0,p[3:])))

    def los_image_galario(self, p):
        '''Version of los_image for galario, no x/y offset, position angle.'''
        img = self.los_image_cutout_(np.append([0.0,0.0,0.0],p))
        image = np.zeros((self.ny, self.nx))
        dx = np.diff(self.rmax[0])[0]
        dy = np.diff(self.rmax[1])[0]
        image[self.ny2-dy//2:self.ny2+dy//2,
              self.nx2-dx//2:self.nx2+dx//2] = img
        return image

    def los_image_galario_axisym(self, p):
        '''Version of los_image for galario, no x/y offset, postion
        angle, or anomaly dependence.
        '''
        return self.los_image_galario(np.append([0.0],p))

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
                        distance, v_sys=0.0):
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
        '''

        x0, y0, pos, anom, inc, tot = p[:6]

        au = 1.496e11

        # get the density cube
        c = self.los_image_cutout_(p, cube=True)

        # cube distances
        x, y, z = np.meshgrid(self.x-x0/self.arcsec_pix,
                              self.y-y0/self.arcsec_pix, self.z)
        r = np.sqrt(x*x + y*y + z*z) * self.arcsec_pix * distance * au

        # get distance along major axis
        t = np.array([[np.cos(np.deg2rad(pos)),-np.sin(np.deg2rad(pos))],
                      [np.sin(np.deg2rad(pos)), np.cos(np.deg2rad(pos))]])
        XY = np.vstack( (x.reshape(-1), y.reshape(-1)) )
        xy = np.matmul(t.T, XY)
        x, y = xy.reshape( (2,) + x.shape )

        # velocities in each pixel for this inclination
        vr = cube.v_rad(y * self.arcsec_pix * distance * au,
                        r, inc, mstar) + v_sys

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
                       distance, v_sys=0.0):
        '''Version of rv_cube_cutout_.
            
        All calls come through here so this is the only routine where
        the cutout offset is subtracted.
        '''
        return self.rv_cube_cutout_(np.append([p[0]-self.x0_arcsec,
                                               p[1]-self.y0_arcsec], p[2:]),
                                    rv_min, dv, n_chan, mstar, distance, v_sys)

    def rv_cube_(self, p, rv_min, dv, n_chan, mstar,
                distance, v_sys=0.0):
        '''Version of rv_cube, full parameters.'''
        rvc = self.rv_cube_cutout(p, rv_min, dv, n_chan, mstar,
                                  distance, v_sys)
        # put this in the image
        c_out = np.zeros((self.ny, self.nx, n_chan))
        c_out[self.ny2+self.rmax[1,0]:self.ny2+self.rmax[1,1],
              self.nx2+self.rmax[0,0]:self.nx2+self.rmax[0,1], :] = rvc
        return c_out

    def rv_cube_axisym(self, p, rv_min, dv, n_chan, mstar,
                       distance, v_sys=0.0):
        '''Version of rv_cube for axisymmetric disks.'''
        return self.rv_cube_(np.append(p[:3],np.append(0.0,p[3:])),
                             rv_min, dv, n_chan, mstar, distance, v_sys)

    def rv_cube_galario_(self, p, rv_min, dv, n_chan,
                         mstar, distance, v_sys=0.0):
        '''Version of los_image_cube for galario, no x/y offset,
        postion angle.
        '''
        rvc = self.rv_cube_cutout_(np.append([0.0,0.0,0.0],p), rv_min, dv,
                                   n_chan, mstar, distance, v_sys)
        c = np.zeros((self.ny, self.nx, n_chan))
        dx = np.diff(self.rmax[0])[0]
        dy = np.diff(self.rmax[1])[0]
        c[self.ny2-dy//2:self.ny2+dy//2,
          self.nx2-dx//2:self.nx2+dx//2, :] = rvc
        return c

    def rv_cube_galario_axisym(self, p, rv_min, dv, n_chan,
                               mstar, distance, v_sys=0.0):
        '''Version of los_image_cube for galario, no x/y offset, postion
        angle, or anomaly dependence.
        '''
        return self.rv_cube_galario_(np.append([0.0],p), rv_min,
                                     dv, n_chan, mstar, distance, v_sys)
