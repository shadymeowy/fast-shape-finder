from setuptools import setup
from setuptools.extension import Extension

cc_flags = ['-O3', '-march=native', '-mtune=native']

modules = [
    Extension(
        name="fast_shape_finder",
        sources=["fast_shape_finder.pyx"],
        extra_compile_args=cc_flags,
        extra_link_args=cc_flags
    ),
]

setup(ext_modules=modules)
