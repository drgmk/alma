import shutil
import os
import argparse
import pickle
import numpy as np
from scipy.stats import binned_statistic_2d
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
import corner

from . import stan_gauss_code

'''
Create M2 (arm-64) conda env with

>CONDA_SUBDIR=osx-arm64 conda create -n stan python=3.11 ipython numpy scipy matplotlib
>conda activate stan
>conda env config vars set CONDA_SUBDIR=osx-arm64
>conda deactivate
>conda activate stan
>conda install -c conda-forge cmdstanpy corner cmdstan>=2.33.1

Get alma package from github and install with pip

>cd alma
>pip install .

This runs ~3x faster than Intel Mac, and >10x faster than osx-64 on an M2.
M2 Pro is nearly 2x faster again.
'''

def alma_stan_gauss():

    # setup
    parser = argparse.ArgumentParser(description='almastan (gaussian)')
    parser.add_argument('-v', dest='visfiles', metavar=('vis1.npy', 'vis2.npy'), nargs='+', required=True,
                        help='numpy save files (u, v, re, im, w, wav, filename)')
    parser.add_argument('-g', dest='g', type=float, nargs=4,
                        metavar=('dra', 'ddec', 'pa', 'inc'),
                        help='geometry parameters')
    parser.add_argument('-p', dest='p', type=float, nargs=4, action='append',
                        metavar=('norm', 'r', 'dr', 'zh'),
                        help='disk parameters')
    parser.add_argument('-o', dest='outdir', metavar='outdir', type=str, default='./',
                        help='folder for output')
    parser.add_argument('--sz', dest='sz', metavar='size arcsec', type=float, default=8.84,
                        help='radius for 1pc  decrement from uv binning')
    parser.add_argument('--sc', dest='sc', metavar='scale', type=float, default=1,
                        help='scale parameters for std ~ 1')
    parser.add_argument('--norm-mul', dest='norm_mul', metavar='norm_mul', type=float, default=10,
                        help='scaling for norm')
    parser.add_argument('--r-mul', dest='r_mul', metavar='r_mul', type=float, default=1,
                        help='scaling for radius')
    parser.add_argument('--star', dest='star', metavar=('flux'),
                        type=float, nargs=1, help='Point source at disk center')
    parser.add_argument('--bg', dest='bg', metavar=('x', 'y', 'f', 'r', 'pa', 'inc'), action='append',
                        type=float, nargs=6, help='resolved background source')
    parser.add_argument('--pt', dest='pt', metavar=('x', 'y', 'f'), action='append',
                        type=float, nargs=3, help='unresolved background source')
    parser.add_argument('-m', dest='metric', metavar='metric.pkl', type=str,
                        help='pickled metric')
    parser.add_argument('--inc-lim', dest='inc_lim', action='store_true', default=False,
                        help="limit range of inclinations")
    parser.add_argument('--r-lim', dest='r_lim', action='store_true', default=False,
                        help="limit range of radii to be +ve")
    parser.add_argument('--dr-lim', dest='dr_lim', action='store_true', default=False,
                        help="limit range of widths to be +ve")
    parser.add_argument('--z-lim', dest='zlim', metavar='zlim', type=float, default=None,
                        help='1sigma upper limit on z/r')
    parser.add_argument('--rew', dest='reweight', action='store_true', default=False,
                        help="Reweight visibilities")
    parser.add_argument('--no-save', dest='save', action='store_false', default=True,
                        help="Don't save model")
    parser.add_argument('--no-pf', dest='pf', action='store_false', default=True,
                        help="Don't run pathfinder")

    args = parser.parse_args()

    outdir = args.outdir.rstrip()

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    visfiles = args.visfiles

    inits = {}
    if args.g:
        inits['dra'], inits['ddec'], inits['pa'], inits['inc'] = args.g
    else:
        inits['dra'], inits['ddec'], inits['pa'], inits['inc'] = 0, 0, 0, 0

    if args.p:
        p = np.array(args.p)
        inits['norm'] = p[:, 0]
        inits['r'] = p[:, 1]
        inits['dr'] = p[:, 2]
        inits['zh'] = p[:, 3]
    else:
        inits['norm'], inits['r'], inits['dr'], inits['zh'] = 0.01, 1, 0.1, 0.05

    if args.star:
        inits['star'] = args.star[0]

    if args.bg:
        bg = np.array(args.bg)
        inits['bgx'] = bg[:, 0]
        inits['bgy'] = bg[:, 1]
        inits['bgn'] = bg[:, 2]
        inits['bgr'] = bg[:, 3]
        inits['bgpa'] = bg[:, 4]
        inits['bgi'] = bg[:, 5]

    if args.pt:
        pt = np.array(args.pt)
        inits['ptx'] = pt[:, 0]
        inits['pty'] = pt[:, 1]
        inits['ptn'] = pt[:, 2]

    # scale parameters to have approx unit standard deviation
    sc = args.sc
    mul = {}
    for p in list(inits.keys()):
        mul[p] = sc
        if p in ['norm', 'bgn', 'ptn', 'star']:
            mul[p] *= args.norm_mul
        if p in ['bgpa', 'bgi']:
            mul[p] /= args.norm_mul
        if p in ['zh'] and args.zlim:
            mul[p] = 1.0

    if args.metric:
        with open(args.metric, 'rb') as f:
            par = pickle.load(f)
            mul = pickle.load(f)
            std = pickle.load(f)
            metric = pickle.load(f)

        for p in par:
            if '_' not in p:
                p_ = p.split('[')[0]
                if '[2]' not in p and '[3]' not in p:
                    mul[p_] = mul[p_] / std[p]

        print(f'scaling from previous std: {mul}')

    data = {'nr': len(inits['norm'])}
    for k in inits.keys():
        inits[k] *= mul[k]
        data[f'{k}_0'] = inits[k]
        data[f'{k}_mul'] = mul[k]

    if args.bg:
        data['nbg'] = len(bg)
    else:
        data['nbg'] = 0

    if args.pt:
        data['npt'] = len(pt)
    else:
        data['npt'] = 0

    stanfile = f'/tmp/alma{str(np.random.randint(100_000))}.stan'

    # load data
    u_ = v_ = Re_ = Im_ = w_ = np.array([])
    for i, f in enumerate(visfiles):
        if '.npy' in f:
            # my format
            u, v, Re, Im, w, wavelength_, ms_file_ = np.load(f, allow_pickle=True)
        elif '.txt' in f:
            # 3 comment lines, then u[m], v[m], Re, Im, w. Line 2 is wave in m
            fh = open(f)
            lines = fh.readlines()
            wavelength_ = float(lines[1].strip().split(' ')[-1])
            u, v, Re, Im, w = np.loadtxt(f, comments='#', unpack=True)
            u /= wavelength_
            v /= wavelength_

        print(f'loading: {f} with nvis:{len(u)}')

        reweight_factor = 2 * len(w) / np.sum((Re**2.0 + Im**2.0) * w)
        print(f' reweighting factor would be {reweight_factor}')
        if args.reweight:
            print(' applying reweighting')
            w *= reweight_factor

        u_ = np.append(u_, u)
        v_ = np.append(v_, v)
        w_ = np.append(w_, w)
        Re_ = np.append(Re_, Re)
        Im_ = np.append(Im_, Im)

    if args.sz > 0:
        # flip negative u if we are averaging
        uneg = u_ < 0
        u_ = np.abs(u_)
        v_[uneg] = -v_[uneg]
        Im_[uneg] *= -1

        def get_duv(R=0.99, size_arcsec=8.84):
            """Return u,v cell size."""
            return 1/(size_arcsec/3600*np.pi/180) * np.sqrt(1/R**2 - 1)

        binsz = get_duv(size_arcsec=args.sz)
        print(f'uv bin: {binsz:.2f}')

        bins = [int(np.max(np.abs(v_))/binsz), int(np.max(np.abs(u_))/binsz)*2]

        u,  _, _, _ = binned_statistic_2d(u_, v_, u_*w_, statistic='sum', bins=bins)
        v,  _, _, _ = binned_statistic_2d(u_, v_, v_*w_, statistic='sum', bins=bins)
        Re, _, _, _ = binned_statistic_2d(u_, v_, Re_*w_, statistic='sum', bins=bins)
        Im, _, _, _ = binned_statistic_2d(u_, v_, Im_*w_, statistic='sum', bins=bins)
        w,  _, _, _ = binned_statistic_2d(u_, v_, w_, statistic='sum', bins=bins)

        # keep non-empty cells
        ok = w != 0
        data['u'] = (u[ok] / w[ok]).flatten()
        data['v'] = (v[ok] / w[ok]).flatten()
        data['re'] = (Re[ok] / w[ok]).flatten()
        data['im'] = (Im[ok] / w[ok]).flatten()
        data['w'] = w[ok].flatten()

    else:
        data['u'] = u_
        data['v'] = v_
        data['re'] = Re_
        data['im'] = Im_
        data['w'] = w_

    data['nvis'] = len(data['u'])
    data['sigma'] = 1/np.sqrt(data['w'])

    print(f"original nvis:{len(u_)}, fitting nvis:{data['nvis']}")

    code = stan_gauss_code.get_code(bg=data['nbg'] > 0, pt=data['npt'] > 0,
                                   star=args.star, gq=False,
                                   inc_lim=args.inc_lim, r_lim=args.r_lim, dr_lim=args.dr_lim,
                                   z_prior=args.zlim)
    with open(stanfile, 'w') as f:
        f.write(code)

    model = CmdStanModel(stan_file=stanfile, cpp_options={'STAN_THREADS': 'TRUE'})
    # model = CmdStanModel(stan_file='/Users/grant/astro/projects/alma/alma/alma/alma_test.stan',
    #                      cpp_options={'STAN_THREADS': 'TRUE'})
    # for k in inits.keys():
    #     inits[k] = np.zeros_like(inits[k])+0.001

    # print(model.exe_info())

    # initial run with pathfinder to estimate parameters and metric
    metric = 'dense'
    if args.pf:
        pf = model.pathfinder(data=data,
                              inits=inits,
                              show_console=False
                              )

        cn = pf.column_names
        ok = ['_' in c and '__' not in c for c in cn]
        fig = corner.corner(pf.draws()[:, ok],
                            titles=np.array(pf.column_names)[ok], show_titles=True)
        fig.savefig(f'{outdir}/corner_pf.pdf')

        ok = ['_' not in c for c in cn]
        metric = {'inv_metric': np.cov(pf.draws()[:, ok].T)}

        for k in inits.keys():
            med = np.median(pf.stan_variable(f'{k}_'), axis=0)
            std = np.std(pf.stan_variable(f'{k}_'), axis=0)
            data[f'{k}_mul'] = 1 / np.mean(std)
            inits[k] = med * data[f'{k}_mul']
            data[f'{k}_0'] = inits[k]
            # print(k, data[f'{k}_mul'], inits[k], (inits[k]/data[f'{k}_mul']))


    # HMC sampling
    fit = model.sample(data=data, chains=6,
                       metric=metric,
                       iter_warmup=1000, iter_sampling=300,
                       inits=inits,
                       show_console=False,
                       refresh=50)

    # fit.save_csvfiles(outdir)
    # shutil.copy(fit.metadata.cmdstan_config['profile_file'], outdir)
    with open(f'{outdir}/metric.pkl', 'wb') as f:
        pickle.dump(fit.column_names, f)
        pickle.dump(mul, f)
        pickle.dump(fit.summary()['StdDev'], f)
        pickle.dump(fit.metric, f)

    df = fit.summary(percentiles=(5, 95))
    print(df[df.index.str.contains('_') == False])
    # print(df.filter(regex='[a-z]_', axis=0))
    # print(fit.diagnose())

    xr = fit.draws_xr()
    for k in inits.keys():
        xr = xr.drop_vars(k)

    _ = az.plot_trace(xr)
    fig = _.ravel()[0].figure
    fig.tight_layout()
    fig.savefig(f'{outdir}/trace.pdf')

    best = {}
    for k in fit.stan_variables().keys():
        if '_' in k and '__' not in k:
            best[k] = np.median(fit.stan_variable(k), axis=0)
    print(f'best: {best}')

    # comment if memory problems ("python killed")
    fig = corner.corner(xr, show_titles=True)
    fig.savefig(f'{outdir}/corner.pdf')

    plt.close('all')

    # save model
    if args.save:
        code = stan_gauss_code.get_code(bg=data['nbg'] > 0, pt=data['npt'] > 0,
                                        star=args.star, gq=True,
                                        inc_lim=args.inc_lim, r_lim=args.r_lim, dr_lim=args.dr_lim,
                                        z_prior=args.zlim)
        with open(stanfile, 'w') as f:
            f.write(code)

        model = CmdStanModel(stan_file=stanfile, cpp_options={'STAN_THREADS': 'TRUE'})

        for k in inits.keys():
            inits[k] = best[f'{k}_'] * mul[k]
        fit1 = model.sample(data=data, inits=inits, fixed_param=True,
                            chains=1, iter_warmup=0, iter_sampling=1)

        for f in visfiles:
            print(f'saving model for {os.path.basename(f)}')
            if '.npy' in f:
                # my format
                u, v, Re, Im, w, _, _ = np.load(f, allow_pickle=True)
            elif '.txt' in f:
                # 3 comment lines, then u[m], v[m], Re, Im, w. Line 2 is wave in m
                fh = open(f)
                lines = fh.readlines()
                wavelength_ = float(lines[1].strip().split(' ')[-1])
                u, v, Re, Im, w = np.loadtxt(f, comments='#', unpack=True)
                u /= wavelength_
                v /= wavelength_

            data['nvis'] = len(u)
            data['u'] = u
            data['v'] = v
            data['re'] = Re
            data['im'] = Im
            data['sigma'] = 1/np.sqrt(w)
            gq = model.generate_quantities(data=data, previous_fit=fit1)
            vis_mod = gq.stan_variables()['vismod_re'] + 1j*gq.stan_variables()['vismod_im']
            np.save(f"{outdir}/{os.path.basename(f).replace('.npy', '-vismod.npy')}", vis_mod)
