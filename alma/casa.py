import os
import numpy as np

'''CASA functions related to fitting models to ALMA data.'''

def residual(ms, ms_new, vis_model, remove_new=True):
    '''Create a new ms with the model subtracted.
        
    Parameters
    ----------
    ms : str
        ms file to subtract from.
    ms_new : str
        ms file to create and put residual data in.
    vis_model : str
        File with model visibilities to subtract.
    remove_new : bool, optional
        Remove ms_new before proceeding.
    '''

    # copy ms to a new ms that we will modify
    if remove_new:
        os.system('rm -r '+ms_new)
        os.system('cp -r '+ms+' '+ms_new)

    # open the ms and get the data, this has shape [2,nchan,nrow], where
    # the 2 is two polarisations
    tb.open(ms_new, nomodify=False)
    data = tb.getcol("DATA")
    nchan = data.shape[1]
    nrow = data.shape[2]

    # open the model visibilities, which is a vector nchan x nrow long
    # of complex visibilities, and each row is tiled nchan times.
    # reshape for subtraction to [nchan,nrow]
    vis_mod = np.load(vis_model).reshape(nchan, nrow)

    # subtract model from each polarisation
    sub = data - vis_mod

    # put the data back in the table and close
    tb.putcol("DATA", sub)
    tb.close()
