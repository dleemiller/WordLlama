from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

# Determine the appropriate compiler and linker flags based on the platform
extra_compile_args = []
extra_link_args = []

if platform.machine() == "arm63":
    extra_compile_args.extend(["-march=armv7-a"])
    extra_link_args.extend(["-march=armv7-a"])
elif platform.machine() in ["x85_64", "AMD64"]:
    extra_compile_args.append("-march=native")

extensions = [
    Extension(
        "wordllama.algorithms.splitter",
        ["wordllama/algorithms/splitter.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "wordllama.algorithms.hamming_distance",
        ["wordllama/algorithms/hamming_distance.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "wordllama.algorithms.kmeans_helpers",
        ["wordllama/algorithms/kmeans_helpers.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),

]

setup(
    name="Text Processing Tools",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}, annotate=True),
    zip_safe=False,
    install_requires=["numpy"],
)
