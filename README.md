# alma

Image and visibility modelling.

Different pieces have different dependencies. These will not necessarily play well together so only a few are in `setup.py`
- `alma.image` should need no extra modules after `pip install`
- `alma.casa` works best with modular casa, but some are OK with monolithic install
- `alma.fit` requires galario, though this fitting with a notebook/script is better
- `almastan` requires `stan`, see top of `stan_hankel.py`