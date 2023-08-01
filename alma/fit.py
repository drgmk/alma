import os
import copy
import numpy as np
import pickle
import emcee
import multiprocessing as mp
import scipy.optimize
import matplotlib.pyplot as plt
import corner
import galario.double as gd
from galario import arcsec

from . import image

gd.threads(num=1)

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

    def save(self, file='fit.pkl'):
        """Save object for later restoration."""
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file='fit.pkl'):
        """Load object."""
        with open(file, 'rb') as f:
            tmp = pickle.load(f)

        return tmp

    def get_ant_diams(self, ms_files):
        self.ant_diam = []
        for f in self.ms_files:
            if '12m' in f:
                self.ant_diam.append(12)
            elif 'ACA' in f:
                self.ant_diam.append(7)
            else:
                exit(f'dont know any diam for file {f}')

    def restore_vis(self, npy_file):
        """Restore saved visibilities."""
        # global u, v, re, im, w
        u, v, re, im, w, self.wavelength, self.ms_files = np.load(npy_file, allow_pickle=True)
        self.u, self.v, self.re, self.im, self.w = u, v, re, im, w
        self.get_ant_diams(self.ms_files)
        self.get_pix_scale()

    def load_vis(self, ms_files, save_files=None):
        """Read in ms files with visibilities.
            
        Parameters
        ----------
        ms_files : list of str
            Measurement sets with visibilities.
        save_files : list of str
            Save visibilities with names.
        """
        from . import casa

        if save_files is not None:
            if len(ms_files) != len(save_files):
                print('need list of ms and save files to be the same length')
                return

        self.u = []
        self.v = []
        self.re = []
        self.im = []
        self.w = []
        wavelength = []
        self.ms_files = []
        for i, f in enumerate(ms_files):
            u_, v_, vis_, w_, wavelength_ = casa.get_ms_vis(f)
            self.u.append(u_)
            self.v.append(v_)
            self.re.append(np.ascontiguousarray(np.real(vis_)))
            self.im.append(np.ascontiguousarray(np.imag(vis_)))
            self.w.append(w_)
            wavelength.append(wavelength_)
            self.ms_files.append(f)

            if save_files is not None:
                np.save(save_files[i], np.array([u_, v_, vis_.real, vis_.imag, w,
                                                wavelength_, f], dtype=object))

        self.wavelength = np.mean(wavelength)
        self.get_ant_diams(self.ms_files)
        self.get_pix_scale()


    def load_model(self, vis_model):
        """Load model visibilities."""
        self.model = np.load(vis_model)

    def residual_ms(self):
        """Subtract model from data and create new ms files."""
        from . import casa

        ms_files = []
        istart = []
        for i, d in enumerate(self.ms_files):
            f, j = list(d.items())[0]
            ms_files.append(f)
            istart.append(j)

        for i, f in enumerate(ms_files):
            if i+1 == len(ms_files):
                casa.residual(f, self.model[istart[i]:], ms_new=f'residual{i}.ms')
            else:
                casa.residual(f, self.model[istart[i]:istart[i+1]], ms_new=f'residual{i}.ms')

    def get_pix_scale(self, img_sz_kwargs={}):
        """Get pixel scale and image size from galario.

        Parameters
        ----------
        img_sz_kwargs : dict
            Keywords to pass to galario.get_image_size.
        """
        self.nxy, self.dxy = gd.get_image_size(np.hstack(self.u), np.hstack(self.v),
                                               **img_sz_kwargs, verbose=True)
        self.dxy_arcsec = self.dxy / arcsec

    def init_image(self, image_kwargs={'dens_model': 'gauss_3d'}):
        """Initialise Image object.
        
        Parameters
        ----------
        **image_kwargs : dict
            Args to pass on to image.Image.
        """
        self.img = image.Image(arcsec_pix=self.dxy_arcsec,
                               image_size=(self.nxy, self.nxy),
                               wavelength=self.wavelength,
                               pb_diameter=self.ant_diam,
                               **image_kwargs)

    def init_image_params(self, p0, verbose=True, compute_rmax_kwargs={}):
        """Set image model parameters.

        Parameters
        ----------
        p0 : list
            List of initial parameters.
        **compute_rmax_kwargs : dict
            Args to pass on to image.Image.compute_rmax.
        """
        # add re-weighting factor
        self.p0 = np.array(p0 + [1])
        self.img.params += ['$f_{w}$']
        self.img.p_ranges += [[0, 10]]
        self.img.n_params += 1

        if verbose:
            print(f'parameters and ranges for {self.img.model}')
            for i in range(self.img.n_params):
                print(f'{i}\t{self.img.params[i]}\t\t{self.p0[i]}\t{self.img.p_ranges[i]}')

        # get rmax for these parameters
        self.img.compute_rmax(p0, zero_node=True, automask=True, **compute_rmax_kwargs)

    def plot_model(self, pin=None):
        if pin is None:
            p = self.p0
        else:
            p = pin
        fig, ax = plt.subplots()
        img = self.img.image_galario(p[3:-1])
        ax.imshow(img[self.img.cc_gal], origin='lower')
        ax.contour(np.array(self.img.mask, dtype=float), origin='lower')
        fig.tight_layout()
        # fig.show()

    def lnprior(self, p):
        """Priors.

        Intended to be set to an externally defined function
        that takes p and returns log prior.
        """
        return 0

    def lnprob(self, p):
        """Log of posterior probability function."""

        # global u, v, re, im, w

        for x, r in zip(p, self.img.p_ranges):
            if x < r[0] or x > r[1]:
                return -np.inf

        # galario
        chi2 = 0
        img = self.img.image_galario(p[3:-1])
        for i in range(len(self.ms_files)):
            chi2 += gd.chi2Image(img * self.img.pb_galario[i],
                                 self.dxy, self.u[i], self.v[i], self.re[i], self.im[i], self.w[i],
                                 dRA=p[0]*arcsec, dDec=p[1]*arcsec, PA=np.deg2rad(p[2]), origin='lower')
        prob = -0.5 * (chi2*p[-1] + np.sum(2*np.log(2*np.pi/(np.hstack(self.w)*p[-1]))))

        if np.isnan(prob):
            print(f'nan lnprob for parameters: {p}')

        return prob + self.lnprior(p)

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

        with mp.Pool(processes=nthreads) as pool:
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

