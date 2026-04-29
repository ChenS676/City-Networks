from setuptools import setup, Extension
import pybind11

compile_args = ["-O3", "-std=c++17", "-fopenmp"]
link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        name="periodic_rw_ext",
        sources=["periodic_rw_bind.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="periodic_rw_ext",
    version="0.1.0",
    ext_modules=ext_modules,
    packages=[],
    py_modules=[],
    zip_safe=False,
)