from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

cc_flags = ['-O3', '-ffast-math', '-march=native', '-mtune=native']

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "fast_shape_finder", ["fast_shape_finder.pyx"],
            extra_compile_args=cc_flags, extra_link_args=cc_flags
        ),
    ],
)