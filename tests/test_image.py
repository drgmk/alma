import numpy as np
import pytest

from .context import alma

def test_image():
    d = alma.image.Image(arcsec_pix=1,
                         image_size=(100,100),
                         wavelength=1e-3
                         )

def test_image_no_primary_beam():
    with pytest.raises(TypeError):
        d = alma.image.Image(arcsec_pix=1, image_size=(100,100))

    d = alma.image.Image(arcsec_pix=1, image_size=(100,100),
                         no_primary_beam=True)
    assert(d.pb is None)


def test_image_dens_extra_par():
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(100,100), wavelength=1e-3,
                         dens_args={'gaussian_scale_height':2})
    assert(d.Dens.gaussian_scale_height==2)

def test_image_face_on_disk():
    sz = 200
    xc = (sz-1)/2.
    x = np.arange(sz)-xc
    xx,yy = np.meshgrid(x,x)
    x0, y0 = 0, 0
    p = [x0,y0,0,0,1,80,5,0.05]
    r = np.sqrt((xx-x0)**2 + (yy-y0)**2)
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(sz,sz), wavelength=1e-3)
    d.compute_rmax(p)
    im = d.image_full(p)
    im_test = d.dens(r,0,0,p[5:]) * d.emit(r,np.inf)
    im_test /= np.sum(im_test)
    assert(np.allclose(im, im_test, atol=0.03*np.max(im)))

def test_image_edge_on_disk():
    sz = 200
    xc = (sz-1)/2.
    x = np.arange(sz)-xc
    xx,yy = np.meshgrid(x,x)
    x0, y0 = 0, 0
    p = [x0,y0,0,90,1,80,5,0.05]
    r = np.sqrt((xx-x0)**2 + (yy-y0)**2)
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(sz,sz), wavelength=1e-3)
    d.compute_rmax(p)
    im = d.image_full(p)
    im = np.sum(im,axis=1)
    im_test = d.dens(r,0,0,p[5:]) * d.emit(r,np.inf)
    im_test = np.sum(im_test,axis=1)
    im_test /= np.sum(im_test)
    assert(np.allclose(im, im_test, atol=0.03*np.max(im)))

def test_image_cube():
    sz = 200
    p = [0,0,20,70,1,1.7,0.5,0.1]
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(sz,sz), wavelength=1e-3)
    d.compute_rmax(p, tol=1e-3, image_full=True)
    d.rv_cube(p)

def test_set_rmax():
    sz = 200
    rmax = 1000
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(sz,sz), wavelength=1e-3)
    with pytest.raises(TypeError):
        d.set_rmax(rmax)

    d.set_rmax(np.array([rmax]))
    assert(np.all(d.rmax[:1]==np.array([sz//2,sz//2]))) 
    assert(d.rmax[2]==rmax)
