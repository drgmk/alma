import numpy as np
import pytest

from .context import alma

def test_image():
    d = alma.image.Image(arcsec_pix=1,
                         image_size=(100,100),
                         wavelength=1e-3
                         )

def test_image_galario_symmetric():
    d = alma.image.Image(arcsec_pix=1, image_size=(100,100), wavelength=1e-3)
    im = d.image_galario([40, 1, 20, 5, 0.2])
    assert(np.sum(np.abs(im[1:,1:] - np.fliplr(im[1:,1:]))) < 1e-15)
    assert(np.sum(np.abs(im[1:,1:] - np.flipud(im[1:,1:]))) < 1e-15)
    assert(np.sum(np.abs(im[1:,1:] - np.rot90(np.rot90(np.fliplr(im[1:,1:]))))) < 1e-15)


def test_image_cutout_offset():
    d = alma.image.Image(arcsec_pix=1, image_size=(200,200), wavelength=1e-3)
    p1 = [0.4, 0.4, 0, 0, 1, 20, 5, 0.2]
    p2 = [2.4, 2.4, 0, 0, 1, 20, 5, 0.2]
    d.compute_rmax(p1)
    im1 = d.image_cutout(p1)
    d.compute_rmax(p2)
    im2 = d.image_cutout(p2)
    assert(np.allclose(im1, im2))


def test_image_primary_beam():
    d = alma.image.Image(arcsec_pix=1, image_size=(100,100))
    assert(d.pb is None)
    assert(d.pb_galario is None)

    d = alma.image.Image(arcsec_pix=1, image_size=(100,100),
                         wavelength=1e3)
    assert(len(d.pb.shape)==2)
    assert(len(d.pb_galario.shape)==2)


def test_image_dens_extra_par():
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(100,100), wavelength=1e-3,
                         dens_args={'gaussian_scale_height':2})
    assert(d.Dens.gaussian_scale_height==2)

def test_image_face_on_disk():
    sz = 400
    xc = (sz-1)/2.
    x = np.arange(sz)-xc
    xx,yy = np.meshgrid(x,x)
    x0, y0 = 0, 0
    p = [x0,y0,0,0,1,80,5,0.05]
    r = np.sqrt((xx-x0)**2 + (yy-y0)**2)
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(sz,sz), wavelength=1e-3)
    d.compute_rmax(p)
    im = d.image(p)
    im_test = d.dens(r,0,0,p[5:]) * d.emit(r,np.inf)
    im_test /= np.sum(im_test)
    assert(np.allclose(im, im_test, atol=0.03*np.max(im)))

def test_image_edge_on_disk():
    sz = 400
    xc = (sz-1)/2.
    x = np.arange(sz)-xc
    xx,yy = np.meshgrid(x,x)
    x0, y0 = 0, 0
    p = [x0,y0,0,90,1,80,5,0.05]
    r = np.sqrt((xx-x0)**2 + (yy-y0)**2)
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(sz,sz), wavelength=1e-3)
    d.compute_rmax(p)
    im = d.image(p)
    im = np.sum(im,axis=1)
    im_test = d.dens(r,0,0,p[5:]) * d.emit(r,np.inf)
    im_test = np.sum(im_test,axis=1)
    im_test /= np.sum(im_test)
    assert(np.allclose(im, im_test, atol=0.03*np.max(im)))

def test_image_cube():
    sz = 400
    p = [0,0,20,70,1,1.7,0.5,0.1]
    rv_min, dv, n_chan, mstar, distance = -10, 1, 20, 1, 20
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(sz,sz), wavelength=1e-3)
    d.compute_rmax(p, tol=1e-3)
    d.rv_cube(p, rv_min, dv, n_chan, mstar, distance)

def test_set_rmax():
    sz = 200
    rmax = 1000
    d = alma.image.Image(arcsec_pix=1, dens_model='gauss_3d',
                         image_size=(sz,sz), wavelength=1e-3)
    with pytest.raises(TypeError):
        d.set_rmax(rmax)
    with pytest.raises(ValueError):
        d.set_rmax(np.array([rmax]))

    rmax = 90
    d.set_rmax(np.array([rmax]))
    for i in [0,1]:
        assert(np.all(d.rmax[i]==np.array([-rmax,rmax])))
    assert(np.all(d.rmax[2]==np.array([-rmax,rmax])))
