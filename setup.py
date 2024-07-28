from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

numpy_include = np.get_include()

extra_compile_args = []
extra_link_args = []

if platform.machine().startswith('arm'):
    if platform.architecture()[0] == '32bit':
        extra_compile_args.extend(["-march=armv7-a", "-mfpu=neon"])
        extra_link_args.extend(["-march=armv7-a", "-mfpu=neon"])
    else:  # 64-bit ARM
        extra_compile_args.extend(["-march=armv8-a+simd"])
        extra_link_args.extend(["-march=armv8-a+simd"])
elif platform.machine() in ["x86_64", "AMD64"]:
    extra_compile_args.extend(["-march=native", "-mpopcnt"])
    extra_link_args.extend(["-march=native", "-mpopcnt"])

extra_compile_args.extend(["-O3", "-ffast-math"])

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
        include_dirs=[numpy_include],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "wordllama.algorithms.kmeans_helpers",
        ["wordllama/algorithms/kmeans_helpers.pyx"],
        include_dirs=[numpy_include],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "wordllama.algorithms.deduplicate_helpers",
        ["wordllama/algorithms/deduplicate_helpers.pyx"],
        include_dirs=[numpy_include],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="Text Processing Tools",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False},
        annotate=True
    ),
    zip_safe=False,
    install_requires=["numpy"],
)

