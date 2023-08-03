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
        self.backend = None
        self.mcmc_savename = 'mcmc.h5'

    def set_outdir(self, outdir):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.out_dir = outdir

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

    def restore_vis(self, npy_files, img_sz_kwargs={}):
        """Restore saved visibilities."""
        # global u, v, re, im, w
        self.u, self.v, self.re, self.im, self.w = [], [], [], [], []
        self.ms_files = []
        wavelength = []
        for f in npy_files:
            u_, v_, Re_, Im_, w_, wavelength_, ms_file_ = np.load(f, allow_pickle=True)
            self.u.append(u_)
            self.v.append(v_)
            self.w.append(w_)
            self.re.append(Re_)
            self.im.append(Im_)
            self.ms_files.append(ms_file_)
            wavelength.append(wavelength_)

        self.wavelength = np.mean(wavelength)
        self.get_ant_diams(self.ms_files)
        self.get_pix_scale(img_sz_kwargs)

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

        self.u, self.v, self.re, self.im, self.w = [], [], [], [], []
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
            self.ms_files.append(os.path.abspath(f))

            if save_files is not None:
                np.save(save_files[i], np.array([u_, v_, vis_.real, vis_.imag, w_,
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

    def init_image(self, p0=None, image_kwargs={'dens_model': 'gauss_3d'},
                   compute_rmax_kwargs={}):
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

        # restore/set parameters
        self.savefile = f'{self.out_dir}/{self.mcmc_savename}'
        if os.path.exists(self.savefile):
            self.backend = emcee.backends.HDFBackend(self.savefile)
            if self.backend.iteration > 0:
                self.init_mcmc()
                self.init_image_params()


        if not hasattr(self, 'mcmc_p0'):
            self.init_image_params(p0=p0)
            self.init_mcmc(nwalkers=int(np.max([os.cpu_count(), self.img.n_params*2])))

        self.compute_rmax(compute_rmax_kwargs)

    def init_image_params(self, p0=None, verbose=True, compute_rmax_kwargs={}):
        """Set image model parameters.

        Parameters
        ----------
        p0 : list
            List of initial parameters.
        **compute_rmax_kwargs : dict
            Args to pass on to image.Image.compute_rmax.
        """
        if not hasattr(self, 'p0'):
            self.p0 = p0

        # add re-weighting factor (if it isn't there already)
        if len(self.p0) == self.img.n_params:
            self.p0 = np.append(self.p0, 1)
        self.img.params += ['$f_{w}$']
        self.img.p_ranges += [[0.9, 1.1]]
        self.img.n_params += 1

        if verbose:
            print(f'parameters and ranges for {self.img.dens_model}')
            for i in range(self.img.n_params):
                print(f'{i}\t{self.img.params[i]}\t\t{self.p0[i]}\t{self.img.p_ranges[i]}')

    def compute_rmax(self, compute_rmax_kwargs):
        # get rmax for these parameters
        self.img.compute_rmax(self.p0, zero_node=True, automask=True, **compute_rmax_kwargs)

    def plot_model(self, par=None):
        if par is None:
            p = self.p0
        else:
            p = par
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
        if self.img.dens_model == 'peri_glow':
            img = ii.image(p[:-1])
            for i in range(len(self.ms_files)):
                chi2 += gd.chi2Image(img * self.img.pb_galario[i],
                                     self.dxy, self.u[i], self.v[i], self.re[i], self.im[i], self.w[i],
                                     origin='lower')
        else:
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

    def init_mcmc(self, nwalkers=None, reset=False):

        self.backend = emcee.backends.HDFBackend(self.savefile)

        if nwalkers is not None:
            self.nwalkers = nwalkers

        if os.path.exists(self.savefile):
            if nwalkers is None:
                self.nwalkers = self.backend.shape[0]
            if self.backend.shape[0] != self.nwalkers:
                reset = True
            if self.backend.iteration > 0:
                ndim = self.backend.shape[1]
                if self.backend.shape[0] > self.nwalkers:
                    self.mcmc_p0 = self.backend.get_chain()[-1, :self.nwalkers, :]
                elif self.backend.shape[0] < self.nwalkers:
                    self.mcmc_p0 = self.backend.get_chain()[-1, np.random.randint(0, self.backend.shape[0], size=self.nwalkers), :]
                else:
                    self.mcmc_p0 = self.backend.get_chain()[-1, :, :]

                self.p0 = np.median(self.mcmc_p0, axis=0)

        if not hasattr(self, 'mcmc_p0'):
            ndim = len(self.p0)
            self.mcmc_p0 = np.array([self.p0 + self.p0*0.01*np.random.randn(ndim) for i in range(self.nwalkers)])
            reset = True

        if reset:
            self.backend.reset(self.nwalkers, len(self.p0))

    def mcmc(self, nsteps=10, nthreads=None):
        """Do the mcmc.
        
        Parameters
        ----------
        nsteps : int
            Number of mcmc steps.
        nthreads : int
            Number of threads.
        """

        if self.nwalkers is None:
            self.nwalkers = self.img.n_params * 2

        with mp.Pool(processes=nthreads) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.img.n_params,
                                            self.lnprob, pool=pool, backend=self.backend)

            self.mcmc_p0, prob, state = sampler.run_mcmc(self.mcmc_p0, nsteps, progress=True)

    def mcmc_plots(self, out_dir=None, savefile=None, post_burn=50):

        if savefile is not None:
            self.backend = emcee.backends.HDFBackend(savefile)
        else:
            if self.backend.iteration == 0:
                print('no data to plot')
                return

        if out_dir is None:
            if self.out_dir is not None:
                out_dir = self.out_dir
            elif self.backend is None:
                out_dir = os.path.abspath(os.path.dirname(self.savefile))
            else:
                print('nowhere to put plots')
                return

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # see what the chains look like, skip a burn in period if desired
        burn = self.backend.iteration - post_burn
        fig,ax = plt.subplots(self.img.n_params+1,2,
                              figsize=(9.5,5), sharex='col', sharey=False)

        for j in range(self.backend.shape[1]):
            ax[-1,0].plot(self.backend.get_log_prob()[:burn, j])
            for i in range(self.img.n_params):
                ax[i, 0].plot(self.backend.get_chain()[:burn, j, i])
                ax[i, 0].set_ylabel(self.img.params[i])

        for j in range(self.backend.shape[0]):
            ax[-1, 1].plot(self.backend.get_log_prob()[burn:, j])
            for i in range(self.img.n_params):
                ax[i, 1].plot(self.backend.get_chain()[burn:, j, i])
                ax[i, 1].set_ylabel(self.img.params[i])

        ax[-1, 0].set_xlabel('burn in')
        ax[-1, 1].set_xlabel('sampling')
        fig.savefig(out_dir+'/chains.png')

        # make the corner plot
        fig = corner.corner(self.backend.get_chain()[burn:, :, :].reshape((-1, self.img.n_params)),
                            labels=self.img.params, show_titles=True)

        fig.savefig(out_dir+'/corner.png')

    def mcmc_medians(self):
        # get the median parameters
        self.mcmc_p = np.median(self.backend.get_chain()[burn:, :, :].reshape((-1, self.img.n_params)),
                           axis=0)
        self.mcmc_std = np.std(self.backend.get_chain()[burn:, :, :].reshape((-1, self.img.n_params)),
                        axis=0)
        print('best fit parameters: {}'.format(self.mcmc_p))

    def save_model_vis(self, out_dir=None):
        if out_dir is None:
            if self.out_dir is not None:
                out_dir = self.out_dir
            elif self.backend is None:
                out_dir = os.path.abspath(os.path.dirname(self.savefile))
            else:
                print('nowhere to put plots')
                return

        # create a copy and recompute the limits to get a full rotated image
        img = copy.deepcopy(self.img)
        img.compute_rmax(self.mcmc_p)
        fig, ax = plt.subplots()
        ax.imshow(img.image(self.mcmc_p)[img.cc], origin='lower')
        fig.savefig(out_dir+'/best.png')

        # save the visibilities for subtraction from the data
        for i in range(len(self.ms_files)):
            if self.img.dens_model == 'peri_glow':
                img = ii.image(self.mcmc_p[:-1])
                vis_mod = gd.sampleImage(img, self.dxy, self.u[i], self.v[i], origin='lower')
            else:
                img = ii.pb_galario[i] * ii.image_galario(self.mcmc_p[3:-1])
                vis_mod = gd.sampleImage(img, self.dxy, self.u[i], self.v[i],
                                         dRA=self.mcmc_p[0]*arcsec, dDec=self.mcmc_p[1]*arcsec,
                                         PA=np.deg2rad(self.mcmc_p[2]),
                                         origin='lower')

            np.save(f'{out_dir}/{os.path.splitext(os.path.basename(self.ms_files[i]))[0]}-mod.npy',
                    vis_mod)
