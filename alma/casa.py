import os
import numpy as np

# test what we are using and import appropriately
try:
    __casashell_state__
    print('Assuming packaged CASA')
except NameError:
    print('Assuming modular CASA')
    import casatools.ms
    import casatools.table
    ms = casatools.ms()
    tb = casatools.table()

'''CASA functions related to fitting models to ALMA data.'''

def residual_standalone(ms, vis_model, tb, datacolumn='CORRECTED_DATA',
             ms_new='residual.ms', remove_new=True,
             model_ms=None, remove_model=True):
    '''Create a new ms with the model subtracted.
        
    Parameters
    ----------
    ms : str
        ms file to subtract from.
    vis_model : str
        File with model visibilities to subtract.
    tb : function
        casa tb function.
    datacolumn : str
        Column to subtract model from.
    ms_new : str, optional
        ms file to create and put residual data in.
    remove_new : bool, optional
        Remove ms_new before proceeding.
    '''

    # copy ms to a new ms that we will modify
    if os.path.exists(ms_new):
        if remove_new:
            os.system('rm -r '+ms_new)

    if not os.path.exists(ms_new):
        os.system('cp -r '+ms+' '+ms_new)
    else:
        raise FileExistsError('file {} exists'.format(ms_new))

    # open the ms and get the data, this has shape [2,nchan,nrow], where
    # the 2 is two polarisations
    tb.open(ms_new, nomodify=False)
    data = tb.getcol(datacolumn)
    flags = tb.getcol('FLAG')
    nchan = data.shape[1]
    nrow = data.shape[2]
    print(f'Opened ms file with {nchan} channels, {nrow} rows')
    print(f'  some ({np.sum(flags)}={np.sum(np.any(flags, axis=(0, 1)))//nchan} rows) data are flagged')

    # some rows will have been flagged
    flagrow = np.sum(np.any(flags, axis=(0, 1)))//nchan
    nrow -= flagrow

    # open the model visibilities, which is a vector nchan x nrow long
    # of complex visibilities, and each row is tiled nchan times.
    # reshape for subtraction to [nchan,nrow]
    vis_mod = np.load(vis_model)
    print('Opened model file with shape {}'.format(vis_mod.shape))
    vis_mod = vis_mod.reshape(nchan, nrow)

    # subtract model from each polarisation
    flags2d = np.logical_or(flags[0, :, :], flags[1, :, :])
    flags2d = np.array([flags2d, flags2d])
    vis_mod = np.array([vis_mod, vis_mod])
    sub = data.copy()
    sub[np.logical_not(flags2d)] = data[np.logical_not(flags2d)] - vis_mod.flatten()
    sub = sub.reshape(2, nchan, nrow+flagrow)

    # put the data back in the table and close
    tb.putcol(datacolumn, sub)
    tb.close()


def residual(ms, vis_model, datacolumn='DATA',
             ms_new=None, remove_new=True, model=False):
    """Create a new ms with the model subtracted.

    Parameters
    ----------
    ms : str
        ms file to subtract from.
    vis_model : str
        File with model visibilities to subtract.
    tb : function
        casa tb function.
    datacolumn : str
        Column to subtract model from.
    ms_new : str, optional
        ms file to create and put residual data in.
    remove_new : bool, optional
        Remove ms_new before proceeding.
    model : bool, optional
        Create ms with only model (no subtraction).
    """

    # copy ms to a new ms that we will modify
    if ms_new is None:
        ms_new = vis_model.replace('.npy', '-residual.ms')
    if os.path.exists(ms_new):
        if remove_new:
            os.system('rm -r '+ms_new)

    if not os.path.exists(ms_new):
        os.system('cp -r '+ms+' '+ms_new)
    else:
        raise FileExistsError('file {} exists'.format(ms_new))

    # open the ms and get the data, this has shape [2,nchan,nrow], where
    # the 2 is two polarisations
    tb.open(ms_new, nomodify=False)
    data = tb.getcol(datacolumn)
    flags = tb.getcol('FLAG')
    nchan = data.shape[1]
    nrow = data.shape[2]
    print(f'Opened ms file with {nchan} channels, {nrow} rows (shape:{data.shape})')
    if np.sum(flags) > 0:
        print(f'  some ({np.sum(flags)}={np.sum(np.any(flags, axis=(0, 1)))//nchan} rows) data are flagged')

    # some rows may have been flagged
    flagrow = np.sum(np.any(flags, axis=(0, 1)))//nchan
    nrow -= flagrow

    # open the model visibilities, which is a vector nchan x nrow long
    # of complex visibilities, and each row is tiled nchan times.
    # reshape for subtraction to [nchan,nrow]
    vis_mod = np.load(vis_model)
    print('Opened model file with shape {}'.format(vis_mod.shape))
    vis_mod = vis_mod.reshape(nchan, nrow)

    # subtract model from each polarisation
    flags2d = np.logical_or(flags[0, :, :], flags[1, :, :])
    flags2d = np.array([flags2d, flags2d])
    vis_mod = np.array([vis_mod, vis_mod])
    sub = data.copy()
    if model:
        sub = np.zeros_like(data)
        sub[np.logical_not(flags2d)] = vis_mod.flatten()
    else:
        sub[np.logical_not(flags2d)] = data[np.logical_not(flags2d)] - vis_mod.flatten()
    sub = sub.reshape(2, nchan, nrow+flagrow)

    # put the data back in the table and close
    tb.putcol(datacolumn, sub)
    tb.close()


def model_ms(ms, vis_model, tb, datacolumn='CORRECTED_DATA',
             ms_new='model.ms', remove_new=True):
    '''Create a new ms with the model.
        
    Parameters
    ----------
    ms : str
        ms file to use as a basis.
    vis_model : str
        File with model visibilities.
    tb : function
        casa tb function.
    datacolumn : str
        Column to put model in (same location as in ms).
    ms_new : str, optional
        ms file to create.
    remove_new : bool, optional
        Remove ms_new before proceeding.
    '''

    # copy ms to a new ms that we will modify
    if os.path.exists(ms_new):
        if remove_new:
            os.system('rm -r '+ms_new)

    if not os.path.exists(ms_new):
        os.system('cp -r '+ms+' '+ms_new)
    else:
        raise FileExistsError('file {} exists'.format(ms_new))

    # open the ms and get the data, this has shape [2,nchan,nrow], where
    # the 2 is two polarisations
    tb.open(ms_new, nomodify=False)
    data = tb.getcol(datacolumn)
    nchan = data.shape[1]
    nrow = data.shape[2]

    # open the model visibilities, which is a vector nchan x nrow long
    # of complex visibilities, and each row is tiled nchan times.
    # reshape for subtraction to [nchan,nrow]
    vis_mod = np.load(vis_model).reshape(nchan, nrow)

    # twice, once for each polarisation
    vis_mod = np.array([vis_mod, vis_mod])

    # put the data in the table and close
    tb.putcol(datacolumn, vis_mod)
    tb.close()


def export_multi_uv_tables(ms, channels, file, tb, split):
    '''Export multiple uv tables for multi-channel data.
    
    Assume ms already has one spw (=0), and that all we need to do
    is select the channels.
    
    While it appears OK, this routine fails to run in casa. It works
    when pasted out and run in an interactive shell.
    '''

    print('This call will probably fail, paste out the code and run it.')
    import uvplot.io
    for i in np.arange(channels[0], channels[1]+1):
        split_args = {'vis':ms,'datacolumn':'data','keepflags':False,
                      'timebin':'20s','spw':'0:{}'.format(i)}
        print('exporting ch {} with args:{}'.format(i,split_args))
        uvplot.io.export_uvtable(file+'-ch{}.txt'.format(i), tb, vis=ms,
                                 split=split,datacolumn='DATA',
                                 split_args=split_args,verbose=True)


def export_ms(msfile, tb, ms, outfile='uv.npy', timescan=False):
    '''Export an ms file to a numpy save file.

    Direct copy of Luca Matra's export.

    Everything is exported, so generally the ms file would already be
    averaged to a small number of channels, e.g. one per spw.

    For this to run smoothly we need to have the same number of channels
    for ALL scans. So no spectral windows with different number of
    channels, otherwise it gets complicated. See:
    https://safe.nrao.edu/wiki/pub/Main/RadioTutorial/BandwidthSmearing.pdf
    to choose how much to smooth a dataset in frequency.

    Errors are taken into account when time averaging in split:
    https://casa.nrao.edu/casadocs/casa-5.1.1/uv-manipulation/time-average

    And when channel averaging:
    https://casa.nrao.edu/casadocs/casa-5.1.1/uv-manipulation/channel-average

    Parameters
    ----------
    ms : str
        ms file to extract.
    outfile : str
        File with model visibilities.
    tb : tb object
        Pass casac.table()
    ms : ms object
        Pass casac.ms()
    '''

#    tb = casac.table()
#    ms = casac.ms()
    cc=2.9979e10 #cm/s

    # Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
    outputfilename=outfile
    msfilename=msfile
    tb.open(msfilename)

    data    = tb.getcol("DATA")
    uvw     = tb.getcol("UVW")
    weight  = tb.getcol("WEIGHT")
    ant1    = tb.getcol("ANTENNA1")
    ant2    = tb.getcol("ANTENNA2")
    flags   = tb.getcol("FLAG")
    spwid   = tb.getcol("DATA_DESC_ID")
    time    = tb.getcol("TIME")
    scan    = tb.getcol("SCAN_NUMBER")
    tb.close()
    if np.any(flags):
        print("Note: some of the data is FLAGGED")
    print("Found data with "+str(data.shape[-1])+" uv points")

    # use CASA ms tools to get the channel/spw info
    ms.open(msfilename)
    spw_info = ms.getspectralwindowinfo()
    nchan = spw_info["0"]["NumChan"]
    npol = spw_info["0"]["NumCorr"]
    ms.close()
    print("with "+str(nchan)+" channels per SPW and "+str(npol)+" polarizations,")

    # Use CASA table tools to get frequencies, which are needed to
    # calculate u-v points from baseline lengths
    tb.open(msfilename+"/SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ")
    rfreq = tb.getcol("REF_FREQUENCY")
    tb.close()
    print(str(freqs.shape[1])+" SPWs and Channel 0 frequency of 1st SPW of "+str(rfreq[0]/1e9)+" GHz")
    print("corresponding to "+str(2.9979e8/rfreq[0]*1e3)+" mm")
    print("Average wavelength is "+str(2.9979e8/np.average(rfreq)*1e3)+" mm")

    print("Datasets has baselines between "+str(np.min(np.sqrt(uvw[0,:]**2.0+uvw[1,:]**2.0)))+" and "+str(np.max(np.sqrt(uvw[0,:]**2.0+uvw[1,:]**2.0)))+" m")

    # Initialize u and v arrays (coordinates in Fourier space)
    uu=np.zeros((freqs.shape[0],uvw[0,:].size))
    vv=np.zeros((freqs.shape[0],uvw[0,:].size))

    # Fill u and v arrays appropriately from data values.
    for i in np.arange(freqs.shape[0]):
        for j in np.arange(uvw.shape[1]):
            uu[i,j]=uvw[0,j]*freqs[i,spwid[j]]/(cc/100.0)
            vv[i,j]=uvw[1,j]*freqs[i,spwid[j]]/(cc/100.0)

    # Extract real and imaginary part of the visibilities at all u-v
    # coordinates, for both polarization states (XX and YY), extract
    # weights which correspond to 1/(uncertainty)^2
    Re_xx = data[0,:,:].real
    Re_yy = data[1,:,:].real
    Im_xx = data[0,:,:].imag
    Im_yy = data[1,:,:].imag
    weight_xx = weight[0,:]
    weight_yy = weight[1,:]

    # Since we don't care about polarization, combine polarization states
    # (average them together) and fix the weights accordingly. Also if
    # any of the two polarization states is flagged, flag the outcome of
    # the combination.
    flags = flags[0,:,:]*flags[1,:,:]
    Re = np.where((weight_xx + weight_yy) != 0, (Re_xx*weight_xx + Re_yy*weight_yy) / (weight_xx + weight_yy), 0.)
    Im = np.where((weight_xx + weight_yy) != 0, (Im_xx*weight_xx + Im_yy*weight_yy) / (weight_xx + weight_yy), 0.)
    wgts = (weight_xx + weight_yy)

    # Find which of the data represents cross-correlation between two
    # antennas as opposed to auto-correlation of a single antenna.
    # We don't care about the latter so we don't want it.
    xc = np.where(ant1 != ant2)[0]

    # Select only cross-correlation data
    time = time[xc]
    spws = spwid[xc]
    scan = scan[xc]
    data_real = Re[:,xc]
    data_imag = Im[:,xc]
    flags = flags[:,xc]
    data_wgts = wgts[xc]
    data_uu = uu[:,xc]
    data_vv = vv[:,xc]
    data_wgts=np.reshape(np.repeat(wgts[xc], uu.shape[0]), data_uu.shape)
    time = np.tile(time, data_uu.shape[0]).reshape(data_uu.shape[0], -1)
    spws = np.tile(spws, data_uu.shape[0]).reshape(data_uu.shape[0], -1)

    # Delete previously used (and not needed) variables (to free up some memory?)
    del Re
    del Im
    del wgts
    del uu
    del vv

    # Select only data that is NOT flagged, this step has the unexpected
    # effect of flattening the arrays to 1d
    data_real = data_real[np.logical_not(flags)]
    data_imag = data_imag[np.logical_not(flags)]
    flagss = flags[np.logical_not(flags)]
    data_wgts = data_wgts[np.logical_not(flags)]
    data_uu = data_uu[np.logical_not(flags)]
    data_vv = data_vv[np.logical_not(flags)]
    time = time[np.logical_not(flags)]
    spws = spws[np.logical_not(flags)]
    # scan = scan[np.logical_not(flags[0])]

    # re-weighting as suggested by Loomis+
    if reweight:
        spw_u = np.unique(spws)
        for s in spw_u:
            ok = spws == s
            wgt_mean = np.mean(data_wgts[ok])
            data_std = np.std(vis[ok])
            rew = (1/data_std**2)/wgt_mean
            logging.info(f're-weighting spw {s}: mean:{wgt_mean}, std:{data_std}')
            logging.info(f're-weighting spw {s} value (1dof): {rew}')
            data_wgts[ok] *= rew

    # Wrap up all the arrays/matrices we need, (u-v coordinates, complex
    # visibilities, and weights for each visibility) and save them all
    # together in a numpy file
    u, v, Re, Im, w = data_uu, data_vv, data_real, data_imag, data_wgts
    
    if timescan:
        np.save(outputfilename, [u, v, Re, Im, w, time, scan])
    else:
        np.save(outputfilename, [u, v, Re, Im, w])


def get_ms_vis(msfilename, xcor=True, acor=False, reweight=True, sort=False):
    """Direct copy of Luca Matra's export.
    https://github.com/dlmatra/miao
    """

    cc = 2.9979e10  # cm/s

    # Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
    tb.open(msfilename)
    data    = tb.getcol("DATA")
    uvw     = tb.getcol("UVW")
    weight  = tb.getcol("WEIGHT")
    ant1    = tb.getcol("ANTENNA1")
    ant2    = tb.getcol("ANTENNA2")
    flags   = tb.getcol("FLAG")
    spwid   = tb.getcol("DATA_DESC_ID")
    time    = tb.getcol("TIME")
    # scan    = tb.getcol("SCAN_NUMBER")
    tb.close()

    if np.any(flags):
        print(f"{msfilename}: some of the data is FLAGGED")

    print("Found data with "+str(data.shape[-1])+" uv points")
    if data.shape[-1] == 0:
        return [], [], [], [], []

    ms.open(msfilename)
    spw_info = ms.getspectralwindowinfo()
    nchan = spw_info[list(spw_info.keys())[0]]['NumChan']
    npol  = spw_info[list(spw_info.keys())[0]]['NumCorr']
    ms.close()
    print("with "+str(nchan)+" channels per SPW and "+str(npol)+" polarizations,")

    # Use CASA table tools to get frequencies, which are needed to
    # calculate u-v points from baseline lengths
    tb.open(msfilename+"/SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ")
    rfreq = tb.getcol("REF_FREQUENCY")
    wave = 2.9979e8/np.average(rfreq)
    tb.close()

    print(str(freqs.shape[1])+" SPWs and Channel 0 frequency of 1st SPW of "+str(rfreq[0]/1e9)+" GHz")
    print("corresponding to "+str(2.9979e8/rfreq[0]*1e3)+" mm")
    print("Average wavelength is "+str(wave*1e3)+" mm")

    if np.max(uvw[0:2, :]) == 0:
        print('No non-zero baselines, returning')
        return [], [], [], [], []

    print("Dataset has baselines between "+str(np.min(np.sqrt(uvw[0, :]**2.0+uvw[1, :]**2.0))) +
                 " and "+str(np.max(np.sqrt(uvw[0, :]**2.0+uvw[1, :]**2.0)))+" m")

    # Initialize u and v arrays (coordinates in Fourier space)
    uu = np.zeros((freqs.shape[0], uvw[0, :].size))
    vv = np.zeros((freqs.shape[0], uvw[0, :].size))

    # Fill u and v arrays appropriately from data values.
    for i in np.arange(freqs.shape[0]):
        for j in np.arange(uvw.shape[1]):
            uu[i, j] = uvw[0, j]*freqs[i, spwid[j]]/(cc/100.0)
            vv[i, j] = uvw[1, j]*freqs[i, spwid[j]]/(cc/100.0)

    # Extract real and imaginary part of the visibilities at all u-v coordinates,
    # for both polarization states (XX and YY), extract weights which correspond
    # to 1/(uncertainty)^2
    Re_xx = data[0,:,:].real
    Im_xx = data[0,:,:].imag
    weight_xx = weight[0,:]
    if npol>=2:
        Re_yy = data[1,:,:].real
        Im_yy = data[1,:,:].imag
        weight_yy = weight[1,:]

        # Since we don't care about polarization, combine polarization states
        # (average them together) and fix the weights accordingly. Also if any
        # of the two polarization states is flagged, flag the outcome of the
        # combination.
        flags = np.logical_or(flags[0,:,:],flags[1,:,:])
        Re = np.where((weight_xx + weight_yy) != 0, (Re_xx*weight_xx + Re_yy*weight_yy) / (weight_xx + weight_yy), 0.)
        Im = np.where((weight_xx + weight_yy) != 0, (Im_xx*weight_xx + Im_yy*weight_yy) / (weight_xx + weight_yy), 0.)
        wgts = (weight_xx + weight_yy)
    else:
        Re=Re_xx
        Im=Im_xx
        wgts=weight_xx
        flags=flags[0,:,:]

    # Find which of the data represents cross-correlation between two
    # antennas as opposed to auto-correlation of a single antenna.
    # We don't care about the latter so we don't want it.
    xc = np.where(ant1 != ant2)[0]
    ac = np.where(ant1 == ant2)[0]
    if xcor:
        if acor:
            xc = np.logical_or(xc, ac)
    elif acor:
        xc = ac

    # Select data
    time = time[xc]
    spws = spwid[xc]
    # scan = scan[xc]
    data_real = Re[:, xc]
    data_imag = Im[:, xc]
    flags = flags[:, xc]
    # data_wgts = wgts[xc]
    data_uu = uu[:, xc]
    data_vv = vv[:, xc]
    data_wgts = np.reshape(np.repeat(wgts[xc], uu.shape[0]), data_uu.shape)
    time = np.tile(time, data_uu.shape[0]).reshape(data_uu.shape[0], -1)
    spws = np.tile(spws, data_uu.shape[0]).reshape(data_uu.shape[0], -1)

    # Select only data that is NOT flagged, this step has the unexpected
    # effect of flattening the arrays to 1d
    data_real = data_real[np.logical_not(flags)]
    data_imag = data_imag[np.logical_not(flags)]
    # flags_ = flags[np.logical_not(flags)]
    data_wgts = data_wgts[np.logical_not(flags)]
    data_uu = data_uu[np.logical_not(flags)]
    data_vv = data_vv[np.logical_not(flags)]
    time = time[np.logical_not(flags)]
    spws = spws[np.logical_not(flags)]
    # scan = scan[np.logical_not(flags[0])]

    time /= (24*60*60)  # to MJD
    vis = data_real + 1j*data_imag

    # sort into time order
    if sort:
        srt = np.argsort(time)
        data_uu = data_uu[srt]
        data_vv = data_vv[srt]
        vis = vis[srt]
        data_wgts = data_wgts[srt]
        time = time[srt]
        spws = spws[srt]

    # re-weighting as suggested by Loomis+
    if reweight:
        spw_u = np.unique(spws)
        for s in spw_u:
            ok = spws == s
            wgt_mean = np.mean(data_wgts[ok])
            data_std = np.std(vis[ok])
            rew = (1/data_std**2)/wgt_mean
            print(f're-weighting spw {s}: mean:{wgt_mean}, std:{data_std}')
            print(f're-weighting spw {s} value (1dof): {rew}')
            data_wgts[ok] *= rew

    return data_uu, data_vv, vis, data_wgts, wave
