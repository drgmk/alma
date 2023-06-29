import os
import copy
import numpy as np
import emcee
import multiprocessing
import scipy.optimize
import matplotlib.pyplot as plt
import corner
import galario.double as gd
from galario import arcsec

from . import image

gd.threads(num=1)

"""Wrappers to help with fitting and automation.

Bascially a copy of the example jupyter notebook script. A call
could look something like this.

```
f = alma.fit.Fit(uv_file='uv.txt')
f.init_image(p0=[0,0,30,50,0.014,2.2,0.3,0.3])
f.optimise()
f.mcmc(nwalk=16, nthreads=4, nsteps=10, burn=5)
```

"""

class Fit(object):
    """Class for fitting and automation."""

    def __init__(self):
        """Get an object for fitting.
            
        Parameters
        ----------
        ms_file : str or list of str
            Measurement set(s) with visibilities.
        """
        self.emcee_pos = None

    def restore_vis(self, npy_file):
        """Restore saved visibilities."""
        self.u, self.v, self.re, self.im, self.w, self.wavelength = np.load(npy_file, allow_pickle=True)
        self.get_pix_scale()

    def load_vis(self, ms_file, save=None):
        """Read in a file with visibilities.
            
        Parameters
        ----------
        ms_file : list of str
            Measurement set with visibilities.
        img_sz_kwargs : dict
            Keywords to pass to galario.get_image_size.
        """

        from . import casa

        self.u = np.array([])
        self.v = np.array([])
        self.w = np.array([])
        vis = np.array([])
        self.wavelength = np.array([])
        for f in ms_file:
            u, v, vv, w, wave = casa.get_ms_vis(f)
            self.u = np.append(self.u, u)
            self.v = np.append(self.v, v)
            self.w = np.append(self.w, w)
            vis = np.append(vis, vv)
            self.wavelength = np.append(self.wavelength, wave)

        self.re = np.ascontiguousarray(np.real(vis))
        self.im = np.ascontiguousarray(np.imag(vis))
        self.wavelength = np.mean(self.wavelength)
        self.get_pix_scale()

        if save is not None:
            np.save(save, [self.u, self.v, self.re, self.im, self.w, self.wavelength])

    def get_pix_scale(self, img_sz_kwargs={}, ):
        """Get pixel scale and image size from galario.

        Parameters
        ----------
        img_sz_kwargs : dict
            Keywords to pass to galario.get_image_size.
        """
        self.nxy, self.dxy = gd.get_image_size(self.u, self.v,
                                               **img_sz_kwargs,
                                               verbose=True)
        self.dxy_arcsec = self.dxy / arcsec

    def init_image(self, p0=None,
                   image_kwargs={'dens_model': 'gauss_3d',},
                   compute_rmax_kwargs={}):
        """Initialise Image object.
        
        Parameters
        ----------
        p0 : list
            List of initial parameters.
        **image_kwargs : dict
            Args to pass on to image.Image.
        **compute_rmax_kwargs : dict
            Args to pass on to image.Image.compute_rmax.
        """
        self.img = image.Image(arcsec_pix=self.dxy_arcsec,
                               image_size=(self.nxy, self.nxy),
                               wavelength=self.wavelength,
                               **image_kwargs)

        # add re-weighting factor
        self.p0 = np.array(p0 + [1])
        self.img.params += ['$f_{w}$']
        self.img.p_ranges += [[0, 10]]
        self.img.n_params += 1

        # get rmax for these parameters
        self.img.compute_rmax(p0, zero_node=True, **compute_rmax_kwargs)

    def lnprob(self, p):
        """Log of posterior probability function."""

        for x,r in zip(p, self.img.p_ranges):
            if x < r[0] or x > r[1]:
                return -np.inf

        # we generate the image with PA = North, origin in lower left,
        # and including primary beam correction
        image = self.img.image_galario(p[3:]) * self.img.pb_galario

        # galario translates and rotates it for us
        chi2 = gd.chi2Image(image,
                            self.dxy, self.u, self.v, self.re, self.im, self.w,
                            origin='lower',
                            dRA=p[0]*arcsec, dDec=p[1]*arcsec, PA=np.deg2rad(p[2]))

        # return the merit function, here we require the weights are normally distributed
        # about the model as a Gaussian with the correct normalisation
        return -0.5 * ( chi2*p[-1] + np.sum(2*np.log(2*np.pi/(self.w*p[-1]))) )
        #     return -0.5 * chi2

    def nlnprob(self, p):
        """Negative log probability, for minimising."""
        return -self.lnprob(p)

    def optimise(self, niter=100):
        """Optimise parameters.

        Parameters
        ----------
        niter : int
            Number of optimisation iterations.
        """
        res = scipy.optimize.minimize(self.nlnprob, self.p0,
                                      method='Nelder-Mead',
                                      options={'maxiter':niter})
        print('Best parameters: {}'.format(res['x']))
        self.p0 = np.array(res['x'])

    def mcmc(self, nwalk=None, nsteps=10, nthreads=None, burn=5,
             out_dir='.', restart=False):
        """Do the mcmc.
        
        Parameters
        ----------
        nwalk : int
            Number of walkers.
        nsteps : int
            Number of mcmc steps.
        nthreads : int
            Number of threads.
        burn : int
            Number of steps to count as burn in.
        out_dir : str
            Where to put results.
        """

        if nwalk is None:
            nwalk = self.img.n_params * 2

        with multiprocessing.Pool(processes=nthreads) as pool:
            sampler = emcee.EnsembleSampler(nwalk, self.img.n_params,
                                            self.lnprob, pool=pool)
            if self.emcee_pos is None or restart:
                self.emcee_pos = [self.p0 + self.p0*0.1*np.random.randn(self.img.n_params) for i in range(nwalk)]
            self.emcee_pos, prob, state = sampler.run_mcmc(self.emcee_pos, nsteps, progress=True)

        model_name = self.img.Dens.model
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # save the chains to file
        np.savez_compressed(out_dir+'/chains-'+model_name+'.npz',
                            sampler.chain, sampler.lnprobability)

        # see what the chains look like, skip a burn in period if desired
        fig,ax = plt.subplots(self.img.n_params+1,2,
                              figsize=(9.5,5), sharex='col', sharey=False)

        for j in range(nwalk):
            ax[-1,0].plot(sampler.lnprobability[j, :burn])
            for i in range(self.img.n_params):
                ax[i, 0].plot(sampler.chain[j, :burn, i])
                ax[i, 0].set_ylabel(self.img.params[i])

        for j in range(nwalk):
            ax[-1, 1].plot(sampler.lnprobability[j, burn:])
            for i in range(self.img.n_params):
                ax[i, 1].plot(sampler.chain[j, burn:, i])
                ax[i, 1].set_ylabel(self.img.params[i])

        ax[-1, 0].set_xlabel('burn in')
        ax[-1, 1].set_xlabel('sampling')
        fig.savefig(out_dir+'/chains-'+model_name+'.png')

        # make the corner plot
        fig = corner.corner(sampler.chain[:, burn:, :].reshape((-1, self.img.n_params)),
                            labels=self.img.params, show_titles=True)

        fig.savefig(out_dir+'/corner-'+model_name+'.png')

        # get the median parameters
        self.p = np.median(sampler.chain[:, burn:, :].reshape((-1, self.img.n_params)),
                           axis=0)
        self.s = np.std(sampler.chain[:, burn:, :].reshape((-1, self.img.n_params)),
                        axis=0)
        print('best fit parameters: {}'.format(self.p))

        # create a copy and recompute the limits to get a full rotated image
        img = copy.deepcopy(self.img)
        img.compute_rmax(self.p)
        fig, ax = plt.subplots()
        ax.imshow(img.image(self.p)[img.cc], origin='lower')
        fig.savefig(out_dir+'/best-'+model_name+'.png')

        # save the visibilities for subtraction from the data
        vis_mod = gd.sampleImage(self.img.pb_galario * self.img.image_galario(self.p[3:]),
                                 self.dxy, self.u, self.v,
                                 dRA=self.p[0]*arcsec, dDec=self.p[1]*arcsec,
                                 PA=np.deg2rad(self.p[2]))
        np.save(out_dir+'/vis-'+model_name+'.npy', vis_mod)

