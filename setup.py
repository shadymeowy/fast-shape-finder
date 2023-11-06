from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

cc_flags = ['-O3', '-march=native', '-mtune=native', '-fopenmp']
# cc_flags = ['-O2']

modules = cythonize([
    Extension(
        "fast_shape_finder", ["fast_shape_finder.pyx"],
        extra_compile_args=cc_flags,
        extra_link_args=cc_flags
    ),
])

setup(
    name='fast-shape-finder',
    version='1.0.0a1',
    description='Finding simple geometric shapes quickly using RANSAC-based algorithms',
    author='Tolga Demirdal',
    url='https://github.com/shadymeowy/fast-shape-finder',
    setup_requires=["cython"],
    install_requires=[],
    ext_modules=modules
)
