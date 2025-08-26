from setuptools import setup, Extension
import pybind11
import os

ext_modules = [
    Extension(
        'sip_cpp',
        ['sip_module.cpp'],
        include_dirs=[pybind11.get_include(), '/usr/include/eigen3'],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3'],
    ),
]

setup(
    name='sip_cpp',
    ext_modules=ext_modules,
)