from setuptools import setup

setup(name='stochbench',
      packages=['stochbench'],
      install_requires=['scipy', 'libsvmdata', 'celer>=0.5.1', 'numpy'])
