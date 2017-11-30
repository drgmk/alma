from setuptools import setup

setup(
    name='alma',
    version='0.1',
    description='tools for modelling alma visibilities',
    url='http://github.com/drgmk/alma',
    author='Grant M. Kennedy',
    author_email='g.kennedy@warwick.ac.uk',
    license='MIT',
    packages=['alma'],
    classifiers=['Programming Language :: Python :: 3'],
    install_requires = ['numpy'],
    zip_safe=False
    )
