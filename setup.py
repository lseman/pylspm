from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
  name = 'plstest',
  ext_modules=[
    Extension('plspm', ['plspm.pyx'])
    ],
  cmdclass = {'build_ext': build_ext},
  include_dirs=[numpy.get_include()]
)