from __future__ import print_function

import numpy as np

class Dens(object):
    '''Define some density functions.
    
    Should peak at or near 1, to aid finding integration box.
    '''

    def __init__(self,model='gauss_3d',gaussian_scale_height=0.05):
        '''Get an object to do density.
        
        Parameters
        ----------
        model : str
            Name of model to use.
        gaussian_scale_height: float
            Scale height to use for fixed-height models.
        '''
        self.gaussian_scale_height = gaussian_scale_height
        self.select(model=model)


    def select(self,model=None, func=None, params=None,
               list_models=False):
        '''Select a density model.
        
        Parameters
        ----------
        model : str
            Name of model to use, one of list returned by list_models.
        func : function
            Custom function to use, takes r, az, el, par.
        params : list
            List of parameter names associated with custom function.
        list_models : bool, optional
            Return list of available models.
        '''

        models = {
            'gauss_3d':{'func':self.gauss_3d,
                        'params':self.gauss_3d_params},
            'gauss_2d':{'func':self.gauss_2d,
                        'params':self.gauss_2d_params},
            'gauss2_3d':{'func':self.gauss2_3d,
                        'params':self.gauss2_3d_params},
            'gauss_3d_test':{'func':self.gauss_3d_test,
                             'params':self.gauss_3d_test_params},
            'power_3d':{'func':self.power_3d,
                        'params':self.power_3d_params},
            'power_top_3d':{'func':self.power_top_3d,
                        'params':self.power_top_3d_params},
            'box_3d':{'func':self.box_3d,
                        'params':self.box_3d_params}
                  }
    
        if list_models:
            return list(models.keys())

        if func is None:
            self.dens = models[model]['func']
            self.params = models[model]['params']
        elif func is not None and params is not None:
            self.dens = func
            self.params = params
        else:
            raise ValueError('incorrect arguments')


    # Gaussian torus and parameters
    gauss_3d_params = ['$r_0$','$\sigma_r$','$\sigma_h$']
    def gauss_3d(self, r, az, el, p):
        '''Gaussian torus.'''
        return np.exp( -0.5*( (r-p[0])/p[1] )**2 ) * \
                    np.exp( -0.5*(r*np.sin(el)/p[2])**2 )

    # Gaussian torus with fixed scale height and parameters
    gauss_2d_params = ['$r_0$','$\sigma_r$']
    def gauss_2d(self, r, az, el, p):
        '''Gaussian torus with fixed scale height.'''
        return np.exp( -0.5*( (r-p[0])/p[1] )**2 ) * \
                np.exp( -0.5*(r*np.sin(el)/self.gaussian_scale_height)**2 )

    # Gaussian in/out torus and parameters
    gauss2_3d_params = ['$r_0$','$\sigma_{r,in}$',
                        '$\sigma_{r,out}$','$\sigma_h$']
    def gauss2_3d(self,r,az,el,p):
        '''Gaussian torus, independent inner and outer sigma.'''
        if r <= p[0]:
            return np.exp( -0.5*( (r-p[0])/p[1] )**2 ) * \
                        np.exp( -0.5*(r*np.sin(el)/p[3])**2 )
        else:
            return np.exp( -( 0.5*(r-p[0])/p[2] )**2 ) * \
                        np.exp( -0.5*(r*np.sin(el)/p[3])**2 )

    # Gaussian torus with wierd azimuthal dependence and parameters
    gauss_3d_test_params = ['$r_0$','$\sigma_r$','$\sigma_h$']
    def gauss_3d_test(self,r,az,el,p):
        '''Gaussian torus with a test azimuthal dependence.'''
        return np.exp( -0.5*( (r-p[0])/p[1] )**2 ) * \
                    np.exp( -0.5*(r*np.sin(el)/p[3])**2 ) * \
                    (az+2*np.pi)%(2*np.pi)

    # Power law torus and parameters
    power_3d_params = ['$r_0$','$p_{in}$','$p_{out}$','$\sigma_h$']
    def power_3d(self,r,az,el,p):
        '''Power law radial profile with Gaussian scale height.'''
        return 1/np.sqrt( (r/p[0])**p[1] + (r/p[0])**(-p[2]) ) * \
                    np.exp( -0.5*(r*np.sin(el)/p[3])**2 )

    # Power top-hat law torus and parameters
    power_top_3d_params = ['$r_0$','$p_{in}$','$p_{out}$',
                           '$\delta_r$','$\sigma_h$']
    def power_top_3d(self,r,az,el,p):
        '''Power law top hat radial profile with Gaussian scale height.'''
        hw = p[3]/2.0
        w2 = p[3]**2
        return np.sqrt( (2+w2) / ( (r/(p[0]+hw))**(2*p[1]) + w2 +
                                   (r/(p[0]-hw))**(-2*p[2]) ) ) * \
                  np.exp( -0.5*(r*np.sin(el)/p[4])**2 )

    # Box torus and parameters
    box_3d_params = ['$r_0$','$\delta_r$','$\delta_h$']
    def box_3d(self,r,az,el,p):
        '''Box torus. assume r,az,el are vectors.'''
        in_i = (r > p[0]-p[1]/2.) & (r < p[0]+p[1]/2.) & \
                   (np.abs(r*np.sin(el)) <= p[2]/2)
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


    def select(self, model=None, func=None, params=None,
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
        list_models : bool, optional
            Return list of available models.
        '''

        models = {
            'blackbody':{'func':self.blackbody,
                        'params':self.blackbody_params}
                  }

        if list_models:
            return list(models.keys())

        if func is None:
            self.emit = models[model]['func']
            self.params = models[model]['params']
        elif func is not None and params is not None:
            self.emit = func
            self.params = params
        else:
            raise ValueError('incorrect arguments')


    # blackbody, no knobs
    blackbody_params = []
    def blackbody(self, r, p):
        '''Blackbody.'''
        return 1.0/r**0.5


class Image(object):
    '''Image generation.'''

    def __init__(self,image_size=None, arcsec_pix=None, rmax_arcsec=None,
                 model='los_image_axisym', emit_model='blackbody',
                 dens_model='gauss_3d', dens_args={},
                 wavelength=None,
                 z_fact=1, verbose=True):
        '''Get an object to make images.

        Parameters
        ----------
        image_size: length 2 tuple (nx, ny)
            Size of output image.
        arcsec_pix: float
            Pixel scale of output image.
        rmax_arcsec: float
            Maximum x,y,z extent of model, for speed.
        model: str
            Integration model to use; includes anomaly or not
        dens_model: str
            Density model to use. Takes some parameters
        dens_args : dict
            Dict of args to be passed to dens.
        emit_model: str
            Emission model to use. Takes no parameters
        wavelength : float
            Wavelength of observations in m, used to create primary beam.
        '''

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
            self.rmax = np.array([self.nx2,self.ny2,self.ny2])
            self.set_rmax(self.rmax)
        else:
            self.set_rmax(np.tile(rmax_arcsec/arcsec_pix,3))

        # set the image model
        self.select(model)
        
        # generate the primary beam
        self.primary_beam(self.wavelength)

        # set the density distribution function
        d = Dens(model=dens_model,**dens_args)
        self.dens = d.dens
        self.dens_params = d.params
        self.n_dens_params = len(self.dens_params)

        # set the emission properties function
        e = Emit(model=emit_model)
        self.emit = e.emit
        self.emit_params = e.params
        self.n_emit_params = len(self.emit_params)

        # paste the params together
        self.params = self.image_params + self.emit_params + self.dens_params
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
                'los_image':{'fit_func':self.los_image,
                             'full_func':self.los_image_full,
                             'params':self.los_image_full_params},
                'los_image_axisym':{'fit_func':self.los_image_axisym,
                                    'full_func':self.los_image_axisym_full,
                                    'params':self.los_image_axisym_params}
                  }

        self.image = models[model]['fit_func']
        self.image_full = models[model]['full_func']
        self.image_params = models[model]['params']
        self.n_image_params = len(self.image_params)


    def compute_rmax(self, p, tol=1e-5, expand=10,
                     image_full=False, radial_only=False):
        '''Figure out model extent to make image generation quicker.
        
        The aim is for non-rotated/shifted image generation to be sped
        up, so self.image is the default. If rmax is the image size (as
        set by default if rmax_arcsec isn't given at istantiation) then
        the radial method is used first.
        
        This routine may be setting the extent used for an entire mcmc
        run, so some wiggle room should be allowed for so that the model
        doesn't move outside the space derived here.
        
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
        image_full : bool, optional
            Use the full (rotated/shifted) image to compute limits.
        radial_only : bool, optional
            Do just a radial calculation of the limits.
        '''
    
        # Work inwards, stopping when density becomes non-zero. Density
        # model is assumed to peak near 1 so absolute tolerance is used.
        if radial_only or self.rmax[0] == self.nx2:
            
            def find_radial():
                for r in np.arange(np.max([self.nx2,self.ny2]),1,-1):
                    for az in np.arange(0,360,30):
                        for el in np.arange(-90,90,15):
                            if self.dens(r*self.arcsec_pix,az,el,
                                         p[self.n_image_params:]) > tol:
                                self.set_rmax(np.tile(r + expand,3))
                                print('radial r_max: {} pix'.\
                                      format(self.rmax[0]))
                                return None
        
            find_radial()

        # get cube of disk, compute limits from that. for a full image,
        # expand the limits to the max allowed first
        if not radial_only:
            if image_full:
                self.set_rmax(np.array([np.max(self.rmax)]))
                cube = self.image_full(p, cube=True)
            else:
                cube = self.image(p[3:], cube=True)
        
            # find model extents, zarray may be higher resolution by
            # factor z_fact, so account for this here
            x_sum = np.sum(cube,axis=(0,1))
            xmax = np.max(
       np.append(np.where( x_sum/np.max(x_sum) > tol )[0]-self.rmax[0],
                 self.rmax[0]-np.where( x_sum/np.max(x_sum) > tol )[0])
                               )
            y_sum = np.sum(cube,axis=(1,2))
            ymax = np.max(
       np.append(np.where( y_sum/np.max(y_sum) > tol )[0]-self.rmax[1],
                 self.rmax[1]-np.where( y_sum/np.max(y_sum) > tol )[0])
                               )
            z_sum = np.sum(cube,axis=(0,2))
            zmax = int( np.max(
       np.append(np.where( z_sum/np.max(z_sum) > tol )[0]-self.rmax[2],
                 self.rmax[2]-np.where( z_sum/np.max(z_sum) > tol )[0])
                               ) / self.z_fact )
            print('model x,y,z extent {}, {}, {}'.format(xmax,ymax,zmax))

            rmax = ( np.array([xmax, ymax, zmax]) + expand ).astype(int)
            self.set_rmax(rmax)


    def set_rmax(self, rmax):
        '''Set rmax and associated things.
            
        Parameters
        ----------
        rmax : np.ndarray
            One or three integers giving the pixel extent of the model.
        '''

        if not isinstance(rmax,np.ndarray):
            raise TypeError('please pass set_rmax an np.ndarray')
        if len(rmax) == 1:
            rmax = np.tile(rmax,3)
        self.rmax = rmax.astype(int)
        self.rmax_arcsec = self.rmax * self.arcsec_pix
        self.crop_size = self.rmax * 2
        self.cc = (slice(self.ny2-self.rmax[1],self.ny2+self.rmax[1]),
                   slice(self.nx2-self.rmax[0],self.nx2+self.rmax[0]))
        y = np.arange(self.crop_size[1]) - (self.crop_size[1]-1)/2.
        z_crop = int(self.crop_size[2] * self.z_fact)
        z = ( np.arange(z_crop) - (z_crop-1)/2. ) / self.z_fact
        self.zarray, self.yarray = np.meshgrid(z, y)


    def primary_beam(self, wavelength, diameter=12.0):
        '''Create an image of the primary beam.
            
        Use Gaussian of FWHM 1.13 lambda/D, based on this:
        https://help.almascience.org/index.php?/Knowledgebase/Article/View/234
        
        Parameters
        ----------
        wavelength : float
            Wavelength of observation in m.
        diameter : float, optional
            Single disk diameter, default is 12m (i.e. ALMA).
        '''
        
        x = np.arange(self.nx) - (self.nx-1)/2.
        y = np.arange(self.ny) - (self.ny-1)/2.
        xx,yy = np.meshgrid(x,y)
        r = np.sqrt(xx**2 + yy**2) * self.rad_pix
        
        self.pb = np.exp(-0.5 * ( r / (1.13*wavelength/diameter/2.35) )**2 )
    

    los_image_full_params = ['$x_0$','$y_0$','$\Omega$','$f$','$i$','$F$']
    def los_image_full(self, p, cube=False):
        '''Return an image of a disk.

        Heavily 'borrows' from zodipic, this is why the rotations are in
        weird directions...
        
        Parameters
        ----------
        p : list
            List of parameters, x0, y0, pos, anom, inc, tot, + emit/dens.
        cube : bool, optional
            Return image cube (y,z,x) for use elsewhere.
        '''
        
        x0, y0, pos, anom, inc, tot = p[:6]
        yarray = self.yarray - y0

        # some geometry, angles are -ve because the
        # rotation matrices go clockwise instead of a.c.w
        c0 = np.cos(np.deg2rad(-pos))
        s0 = np.sin(np.deg2rad(-pos))
        c1 = np.cos(np.deg2rad(inc))
        s1 = np.sin(np.deg2rad(inc))
        c2 = np.cos(np.deg2rad(-anom)+np.pi/2)
        s2 = np.sin(np.deg2rad(-anom)+np.pi/2)
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

        # cube method, no quicker
        if cube:
            x = np.arange(self.crop_size[0]) + 0.5 - self.rmax[0] - x0
            x3 = c0c1c2_s0s2*x + trans1[:,:,None]
            y3 = c0c1s2_s0c2*x + trans2[:,:,None]
            z3 = -c0s1*x + trans3[:,:,None]
                
            # get the spherical polars
            rxy2 = x3**2 + y3**2
            rxy = np.sqrt(rxy2)
            r = np.sqrt(rxy2 + z3**2) * self.arcsec_pix
            az = np.arctan2(y3,x3)
            el = np.arctan2(z3,rxy)

            # the density, dimensions are y, z, x
            cube = self.emit(r,p[slice(6,6+self.n_emit_params)]) * \
                    self.dens(r,az,el,p[6+self.n_emit_params:])

            # if we were to put this in the image
#            image[self.ny2-self.rmax[1]:self.ny2+self.rmax[1],
#                  self.nx2-self.rmax[0]:self.nx2+self.rmax[0]] = np.sum(cube,axis=1)

            return cube

        # go through slice by slice and get the flux along an x column
        # attempts to speed this up using multiprocess.Pool failed
        else:

            image = np.zeros((self.ny, self.nx))

            for i in np.arange(self.crop_size[0]):

                x = i + 0.5 - self.rmax[0] - x0

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
                        self.dens(r,az,el,p[6+self.n_emit_params:])

                # put this in the image
                image[self.ny2-self.rmax[1]:self.ny2+self.rmax[1],
                      self.nx2-self.rmax[0]+i] = np.sum(layer,axis=1)

        return tot * image / np.sum(image)


    def los_image(self, p, cube=False):
        '''Version of los_image_full, no x/y offset, postion angle.'''
        return self.los_image_full(np.append([0.0,0.0,0.0],p),
                                   cube=cube)


    los_image_axisym_params = ['$x_0$','$y_0$','$\Omega$','$i$','$F$']
    def los_image_axisym_full(self, p, cube=False):
        '''Version of los_image_full, no anomaly dependence in dens.'''
        return self.los_image_full(np.append(p[:3],np.append(0.0,p[3:])),
                                   cube=cube)


    def los_image_axisym(self, p, cube=False):
        '''Version of los_image_full, no x/y offset, postion
        angle, or anomaly dependence in dens.
        '''
        return self.los_image_full(np.append([0.0,0.0,0.0,0.0],p),
                                   cube=cube)
