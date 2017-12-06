
# coding: utf-8

# # Fit an image model to ALMA uv data
# The data here have been heavily cut down to keep the size manageable. A typical size might be more like 100MB. This notebook takes about 4 minutes to execute on a pretty gutless macbook.

# In[1]:


import os
import copy
import numpy as np
import emcee
import scipy.optimize
import matplotlib.pyplot as plt
import corner
import galario.double as gd
from galario import arcsec
from uvplot import UVTable

import alma.image

#get_ipython().run_line_magic('matplotlib', 'notebook')


# ## creating the uv table
# This can be exported from the measurement set (ms) file using uvplot from wihtin CASA. The ms file is first created from the full ms by keeping only the target of interest and the intents that observe it, and averaging the CO J=3-2 spectral window down to 128 channels, the same as the other three. The column with the data after this step is 'data'.
# 
# The version of uvplot used was modified to export all data, not just the first channel. The commands are then something like below. The spw are not specified because we want all spectral windows. We specify and keep the outputvis file, as this is what we will subtract the results of the modelling from to make a residual plot.
# 
# ```python
# import uvplot
# uvplot.uvtable.export_uvtable('uv-w32-t20.txt', tb, vis='calibrated.split.cont.ms',
#                               split=split, keepms=True, split_args={
#                                                     'vis':'calibrated.split.cont.ms',
#                                                     'outputvis':'calibrated.split.cont.w32-t20.ms',
#                                                     'timebin':'20s','width':32,
#                                                     'datacolumn':'data'
#                                                                     }
#                              )
# ```

# In[2]:


# import the data, this assumes we're getting the output from uvplot
uv_file = 'hr4796-uv-spw0-w64-t30s.txt'
u, v, Re, Im, w = np.require( np.loadtxt(uv_file, unpack=True),requirements=["C_CONTIGUOUS"])

# meaning we can get the mean wavelength like so
with open(uv_file) as f:
    _ = f.readline()
    tmp = f.readline()

wavelength = float(tmp.strip().split('=')[1])
print('wavelength is {} mm'.format(wavelength*1e3))
    
u /= wavelength
v /= wavelength

# re-weight so that chi^2 for null model is 1
reweight_factor = np.sum( ( Re**2.0 + Im**2.0) * w ) / len(w)
print('reweighting factor is {}'.format(reweight_factor))
w /= reweight_factor


# In[3]:


# set image properties, can cut dxy down for speed (at the cost of reduced accuracy)
nxy, dxy = gd.get_image_size(u, v, verbose=True)
dxy *= 2
dxy_arcsec = dxy / arcsec

xc = (nxy-1)/2.
x = np.arange(nxy)-xc
xx,yy = np.meshgrid(x,x)


# In[4]:


# decide what model we want to use, and where we will put the results
model_name = 'gauss_3d'
if not os.path.exists(model_name):
    os.mkdir(model_name)


# In[5]:


# make the image object
ii = alma.image.Image(arcsec_pix=dxy_arcsec, image_size=(nxy, nxy),
                      dens_model=model_name, z_fact=1, wavelength=wavelength)


# In[6]:


# parameters, got somehow...
common_p = [-0.002, -0.052, 26.0, 78, 0.014, 1.]
if model_name == 'gauss_3d':     p0 = common_p + [0.1, 0.1]
if model_name == 'gauss_2d':     p0 = common_p + [0.1]
if model_name == 'box_3d':       p0 = common_p + [0.3, 0.3]
if model_name == 'power_3d':     p0 = common_p + [20, 20, 0.1]
if model_name == 'power_top_3d': p0 = common_p + [20, 20, 0.1, 0.1]

# parameter space domain
common_p_rng = [[-0.1,0.1],[-0.1,0.1],[-180,180],[0.,90],[0.,1.],[0.,3.]]
if model_name == 'gauss_3d':     p_ranges = common_p_rng + [[0.01,1.], [0.01,1.]]
if model_name == 'gauss_2d':     p_ranges = common_p_rng + [[0.01,1.]]
if model_name == 'box_3d':       p_ranges = common_p_rng + [[0.01,1.], [0.01,1.]]
if model_name == 'power_3d':     p_ranges = common_p_rng + [[1,50], [1,50], [0.01,1.]]
if model_name == 'power_top_3d': p_ranges = common_p_rng + [[1,50], [1,50], [0.01,1.], [0.01,1.]]

print('parameters and ranges for {}'.format(model_name))
for i in range(ii.n_params):
    print('{}\t{}\t{}'.format(p0[i],p_ranges[i],ii.params[i]))


# In[7]:


# set rmax based on these params, tolerance in compute_rmax might be
# varied if the crop size turns out too large
ii.compute_rmax(p0, tol=1e-2, expand=5)

# this gives an idea of how long an mcmc might take
#get_ipython().run_line_magic('timeit', 'ii.image_full(p0)')

# show an image and the primary beam
im = ii.image(p0[3:])
fig,ax = plt.subplots(1,2, figsize=(9.5,5))
ax[0].imshow(im[ii.cc], origin='bottom')
ax[1].imshow(ii.pb[ii.cc], origin='bottom')


# In[8]:


def lnpostfn(p):
    """ Log of posterior probability function """

    for i in range(len(p)):
        if p[i] < p_ranges[i][0] or p[i] > p_ranges[i][1]:
            return -np.inf

    # we generate the image with PA = North, including primary beam correction
    image = ii.image(p[3:]) * ii.pb
    
    # galario  translates and rotates it for us
    chi2 = gd.chi2Image(image, dxy, u, v, Re, Im, w,
                        dRA = p[0]*arcsec, dDec = p[1]*arcsec, PA = np.deg2rad(p[2]) )
    return -0.5 * chi2

nlnpostfn = lambda p: -lnpostfn(p)


# In[9]:


# get a best fit to estimate mcmc starting params
res = scipy.optimize.minimize(nlnpostfn, p0, method='Nelder-Mead',
                             options={'maxiter':100})
print(res['x'])
p0 = np.array(res['x'])

# show the image, before translation and rotation
im = ii.image(p0[3:])
fig,ax = plt.subplots()
ax.imshow(im[ii.cc],origin='bottom')


# In[10]:


# set up and run mcmc fitting
ndim = len(p_ranges)        # number of dimensions
nwalkers = 16               # number of walkers
nsteps = 100               # total number of MCMC steps
nthreads = 8                # CPU threads that emcee should use

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpostfn, threads=nthreads)

# initialize the walkers with an ndim-dimensional Gaussian ball
pos = [p0 + p0*0.1*np.random.randn(ndim) for i in range(nwalkers)]

# execute the MCMC
pos, prob, state = sampler.run_mcmc(pos, nsteps)


# In[11]:


# see what the chains look like, skip a burn in period if desired
burn = 50
fig,ax = plt.subplots(ndim+1,2,figsize=(9.5,5),sharex='col',sharey=False)

for j in range(nwalkers):
    ax[-1,0].plot(sampler.lnprobability[j,:burn])
    for i in range(ndim):
        ax[i,0].plot(sampler.chain[j,:burn,i])
        ax[i,0].set_ylabel(ii.params[i])

for j in range(nwalkers):
    ax[-1,1].plot(sampler.lnprobability[j,burn:])
    for i in range(ndim):
        ax[i,1].plot(sampler.chain[j,burn:,i])
        ax[i,1].set_ylabel(ii.params[i])

ax[-1,0].set_xlabel('burn in')
ax[-1,1].set_xlabel('sampling')
fig.savefig(model_name+'/chains-'+model_name+'.png')


# In[12]:


# make the corner plot
fig = corner.corner(sampler.chain[:,burn:,:].reshape((-1,ndim)), labels=ii.params,
                    show_titles=True)

fig.savefig(model_name+'/corner-'+model_name+'.png')


# In[13]:


# get the median parameters
p = np.median(sampler.chain[:,burn:,:].reshape((-1,ndim)),axis=0)
s = np.std(sampler.chain[:,burn:,:].reshape((-1,ndim)),axis=0)
print(p)
print(s)
p=p0

# recompute the limits for the full rotated image
ii.compute_rmax(p, image_full=True)

fig,ax = plt.subplots()
ax.imshow(ii.image_full(p)[ii.cc], origin='bottom')
fig.savefig(model_name+'/best-'+model_name+'.png')


# In[14]:


# save the chains to file
np.savez_compressed(model_name+'/chains-'+model_name+'.npz', sampler.chain, sampler.lnprobability)


# In[15]:


# save the visibilities for subtraction from the data
vis_mod = gd.sampleImage(ii.pb * ii.image(p[3:]), dxy, u, v, dRA = p[0]*arcsec, 
                        dDec = p[1]*arcsec, PA = np.deg2rad(p[2]) )
np.save(model_name+'/vis-'+model_name+'.npy', vis_mod)


# ## Creating a map of the residuals
# This must be done within CASA. First the script 'residual' is run to subtract the model visibilities created above from the ms we made at the top.
# ```python
# residual('../../../data/alma/hd181327-c3/calibrated.split.cont.w32-t20.ms','tmp.ms','vis-gauss_2d.npy')
# ```
# Then this is imaged using clean, with something like:
# ```python
# clean(vis='tmp.ms',imagename='tmp',imsize=[256,256],cell='0.1arcsec',interactive=True)
# ```
# This image can be checked out with viewer(), or saved to a FITS image.

# In[16]:


# make a uv distance plot
uvbin_size = 30e3     # uv-distance bin [wle]

uv = UVTable(filename=uv_file, wle=wavelength)
uv.apply_phase(p[0]*arcsec, p[1]*arcsec)
uv.deproject(np.deg2rad(p[2]), np.deg2rad(p[1]))

uv_mod = UVTable(uvtable=(u*wavelength, v*wavelength, np.real(vis_mod), np.imag(vis_mod), w), wle=wavelength)
uv_mod.apply_phase(p[0]*arcsec, p[1]*arcsec)
uv_mod.deproject(np.deg2rad(p[2]), np.deg2rad(p[1]))

axes = uv.plot(label='Data', uvbin_size=uvbin_size)
uv_mod.plot(label='Model', uvbin_size=uvbin_size, axes=axes, yerr=False, linestyle='-', color='r')
axes[0].figure.savefig(model_name+'/uvplot-'+model_name+'.png')

