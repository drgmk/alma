import os
import numpy as np
import uvplot.io

'''CASA functions related to fitting models to ALMA data.'''

def residual(ms, vis_model, tb, datacolumn='CORRECTED_DATA',
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
    nchan = data.shape[1]
    nrow = data.shape[2]
    print('Opened ms file with {} channels, {} rows'.format(nchan,nrow))

    # open the model visibilities, which is a vector nchan x nrow long
    # of complex visibilities, and each row is tiled nchan times.
    # reshape for subtraction to [nchan,nrow]
    vis_mod = np.load(vis_model)
    print('Opened model file with shape {}'.format(vis_mod.shape))
    vis_mod = vis_mod.reshape(nchan, nrow)

    # subtract model from each polarisation
    sub = data - vis_mod

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
    for i in np.arange(channels[0], channels[1]+1):
        split_args = {'vis':ms,'datacolumn':'data','keepflags':False,
                      'timebin':'20s','spw':'0:{}'.format(i)}
        print('exporting ch {} with args:{}'.format(i,split_args))
        uvplot.io.export_uvtable(file+'-ch{}.txt'.format(i), tb, vis=ms,
                                 split=split,datacolumn='DATA',
                                 split_args=split_args,verbose=True)
