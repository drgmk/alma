import numpy as np
import los_funcs

class Dens(object):
    '''Define some density functions.
    
    Should in general peak at or near 1, to aid finding integration box.
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
        self.select(model)

    def select(self,model):

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
            'box_3d':{'func':self.box_3d,
                        'params':self.box_3d_params}
                  }

        self.dens = models[model]['func']
        self.params = models[model]['params']

    # Gaussian torus and parameters
    gauss_3d_params = ['$r_0$','$\sigma_r$','$\sigma_h$']
    def gauss_3d(self, r, az, el, p):
#        '''Gaussian torus.'''
        return np.exp( -( 0.5*(r-p[0])/p[1] )**2 ) * \
                    np.exp( -0.5*(r*np.sin(el)/p[2])**2 )

    # Gaussian torus with fixed scale height and parameters
    gauss_2d_params = ['$r_0$','$\sigma_r$']
    def gauss_2d(self, r, az, el, p):
        '''Gaussian torus with fixed scale height.'''
        return np.exp( -( 0.5*(r-p[0])/p[1] )**2 ) * \
                np.exp( -0.5*(r*np.sin(el)/self.gaussian_scale_height)**2 )

    # Gaussian in/out torus and parameters
    gauss2_3d_params = ['$r_0$','$\sigma_{r,in}$',
                        '$\sigma_{r,out}$','$\sigma_h$']
    def gauss2_3d(self,r,az,el,p):
        '''Gaussian torus, independent inner and outer sigma.'''
        if r <= p[0]:
            return np.exp( -( 0.5*(r-p[0])/p[1] )**2 ) * \
                        np.exp( -0.5*(r*np.sin(el)/p[2])**2 )
        else:
            return np.exp( -( 0.5*(r-p[0])/p[2] )**2 ) * \
                        np.exp( -0.5*(r*np.sin(el)/p[3])**2 )

    # Gaussian torus with wierd azimuthal dependence and parameters
    gauss_3d_test_params = ['$r_0$','$\sigma_r$','$\sigma_h$']
    def gauss_3d_test(self,r,az,el,p):
        '''Gaussian torus with a test azimuthal dependence.'''
        return np.exp( -( 0.5*(r-p[0])/p[1] )**2 ) * \
                    np.exp( -0.5*(r*np.sin(el)/p[2])**2 ) * \
                    (az+2*np.pi)%(2*np.pi)

    # Power law torus and parameters
    power_3d_params = ['$r_0$','$p_{in}$','$p_{out}$','$\sigma_h$']
    def power_3d(self,r,az,el,p):
        '''Power law radial profile with Gaussian scale height.'''
        return 1/np.sqrt( (r/p[0])**p[1] + (r/p[0])**(-p[2]) ) * \
                    np.exp( -0.5*(r*np.sin(el)/p[2])**2 )

    # Box torus and parameters
    box_3d_params = ['$r_0$','$\delta_r$','$\delta_h$']
    def box_3d(self,r,az,el,p):
        '''Box torus. assume r,az,el are vectors.'''
        dens = np.zeros(len(r))
        in_i = (r > p[0]-p[1]/2.) & (r < p[0]+p[1]/2.) & \
                   (np.abs(r*np.sin(el)) <= p[2]/2)
        dens[in_i] = 1.0
        return dens


class Emit(object):
    '''Define some emission property functions.'''

    def blackbody(r):
        '''Blackbody. No normalisation.'''
        return 1.0/r**0.5


class Image(object):
    '''Image generation.'''

    def __init__(self,image_size=None, arcsec_pix=None, rmax_arcsec=None,
                 model='los_image_axisym',
                 dens_model='gauss_3d', dens_args={},
                 emit_model='blackbody',
                 verbose=True):
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
        emit_model: str
            Emission model to use. Takes no parameters
        '''

        self.image_size = image_size
        self.model = model
        self.emit_model = emit_model
        self.dens_model = dens_model
        self.arcsec_pix = arcsec_pix

        # set fixed some things needed to make images
        self.nx, self.ny = self.image_size
        self.nx2 = self.nx // 2
        self.ny2 = self.ny // 2
        if rmax_arcsec is not None:
            self.set_rmax(rmax_arcsec)

        # set the image model
        self.select(model)

        # set the density distribution function
        d = Dens(model=dens_model,**dens_args)
        self.dens = d.dens
        self.dens_params = d.params
        self.n_dens_params = len(self.dens_params)

        # paste the params together
        self.params = self.image_params + self.dens_params

        # set the emission properties function
        if emit_model == 'blackbody':
            self.emit = Emit.blackbody

        # say something about the model
        if verbose:
            print('model:{} with density:{} and emit:{}'.\
                    format(model,dens_model,emit_model))
            print('parameters are {}'.format(self.params))


    def select(self,model):
        '''Select the model we want to use.'''

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


    def compute_rmax(self, p, tol=1e-5):
        '''Figure out model extent to make image generation quicker.
        
        Work inwards, stopping when density becomes non-zero.
        '''
    
        for r in self.arcsec_pix*np.arange(np.max([self.nx2,self.ny2]),1,-1):
            for az in np.arange(0,360,30):
                for el in np.arange(-90,90,15):
                    if self.dens(r,az,el,p[self.n_image_params:]) > tol:
                        self.set_rmax(r)
                        print('found r_max: {} ({} pix)'.format(r,self.rmax))
                        return None


    def set_rmax(self,rmax_arcsec):
        '''Set rmax and associated things.'''

        self.rmax_arcsec = rmax_arcsec
        self.rmax = int(rmax_arcsec / self.arcsec_pix)
        self.crop_size = self.rmax * 2
        a = np.arange(self.crop_size) - (self.crop_size-1)/2.
        self.zarray, self.yarray = np.meshgrid(a, a)


    los_image_full_params = ['$x_0$','$y_0$','$\Omega$','$f$','$i$','$F$']
    def los_image_full(self,p,save_bounds=False):
        '''Return an image of a disk.

        Heavily 'borrows' from zodipic.
        '''
        
        x0, y0, pos, anom, inc, tot = p[:6]
        image = np.zeros((self.ny, self.nx))
        yarray = self.yarray - y0

        # some geometry, angles are -ve because the
        # rotation matrices go clockwise instead of a.c.w
        c0 = np.cos(np.deg2rad(-pos))
        s0 = np.sin(np.deg2rad(-pos))
        c1 = np.cos(np.deg2rad(inc))
        s1 = np.sin(np.deg2rad(inc))
        c2 = np.cos(np.deg2rad(-anom+np.pi/2))
        s2 = np.sin(np.deg2rad(-anom+np.pi/2))
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

        # go through slice by slice and get the flux along an x column
        for i in np.arange(self.crop_size):

            x = i + 0.5 - self.rmax - x0

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
            layer = self.emit(r) * self.dens(r,az,el,p[6:])

            # put this in the image
            image[self.ny2-self.rmax:self.ny2+self.rmax,
                  self.nx2-self.rmax+i] = np.sum(layer,axis=1)
            
        return tot * image / np.sum(image)


    def los_image(self,p):
        '''Cut down version of los_image_full, not x/y offset, postion
        angle.
        '''
        return self.los_image_full(np.append([0.0,0.0,0.0],p))


    los_image_axisym_params = ['x_0','y_0','$\Omega$','$i$','F']
    def los_image_axisym_full(self,p):
        '''Version of los_image_full, but no anomaly dependence in 
        density function.
        '''
        return self.los_image_full(np.append(p[:3],np.append(0.0,p[3:])))


    def los_image_axisym(self,p):
        '''Cut down version of los_image_full, not x/y offset, postion
        angle, or anomaly dependence in density function.
        '''
        return self.los_image_full(np.append([0.0,0.0,0.0,0.0],p))
