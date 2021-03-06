import numpy as np

from .context import alma

def test_emit():
    e = alma.image.Emit()

def test_list_models():
    e = alma.image.Emit()
    assert(isinstance(e.select(list_models=True),list))

def test_blackbody():
    e = alma.image.Emit(model='blackbody')
    assert(np.allclose(e.emit(1,[1,100],1000),
                       e.emit(1,[1,100],2000)*4))
    assert(np.allclose(e.emit(1,[1,100],1000),
                       e.emit(1,[1,200],1000)/2))
    assert(np.allclose(e.emit(1,[1,100],1000),
                       e.emit(0.5,[1,100],1000)/np.sqrt(2)))

def test_rj_tail():
    r = 1 + np.arange(10)
    t = 1 / r**0.5
    e = alma.image.Emit(model='rj_tail')
    assert(np.all(np.equal(t,e.emit(r,np.inf))))

def test_other_emit_function():
    r = 1 + np.arange(10)
    f = lambda x,p: 1 / x**0.4
    t = f(r,np.inf)
    e = alma.image.Emit()
    e.select(func=f, params=[])
    assert(np.all(np.equal(t,e.emit(r,np.inf))))

def test_other_emit_function_with_params():
    r = 1 + np.arange(10)
    f = lambda x,p: p / x**0.4
    t = f(r,5)
    e = alma.image.Emit()
    e.select(func=f, params=['exp','norm'])
    assert(np.all(np.equal(t,e.emit(r,5))))
