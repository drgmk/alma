import numpy as np

from .context import alma

def test_dens():
    d = alma.image.Dens()

def test_list_models():
    e = alma.image.Dens()
    assert(isinstance(e.select(list_models=True),list))

def test_dens_types():
    d = alma.image.Dens()
    types = d.select(list_models=True)
    for t in types:
        d.select(t)
        npar = len(d.params)
        d.dens(np.array(1),np.array(0),np.array(0),
               np.arange(npar)+1.0)

def test_dens_gauss_2d():
    r = np.arange(50)
    r0 = 25.
    w = 4.
    de = np.exp(-0.5*((r-r0)/w)**2)
    d = alma.image.Dens(model='gauss_2d')
    assert(np.all(np.equal(de,d.dens(r,0,0,[r0,w]))))

def test_other_dens_function():
    r = np.arange(50)
    r0 = 25.
    w = 4.
    f = lambda r,r0,w: np.exp(-0.5*((r-r0)/w)**3)
    de = f(r,r0,w)
    d = alma.image.Dens(model='gauss_2d')
    d.select(func=f, params=['r0','w'])
    assert(np.all(np.equal(de,d.dens(r,r0,w))))
