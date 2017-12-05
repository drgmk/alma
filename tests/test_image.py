import numpy as np

from .context import alma

def test_image():
    d = alma.image.Image(arcsec_pix=1,
                         image_size=(100,100),
                         wavelength=1e-3
                         )
