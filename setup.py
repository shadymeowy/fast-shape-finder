from setuptools import setup
from setuptools.extension import Extension
import platform

if platform.system() == 'Linux':
    cc_flags = ['-O3']
elif platform.system() == 'Windows':
    cc_flags = ['/O2']
else:
    cc_flags = ['-O3']

modules = [
    Extension(
        name="fast_shape_finder",
        sources=["fast_shape_finder.pyx"],
        extra_compile_args=cc_flags,
        extra_link_args=cc_flags
    ),
]

setup(ext_modules=modules)
