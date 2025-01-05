from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

numpy_include = np.get_include()

extra_compile_args = ["-O3", "-ffast-math"]
extra_link_args = []
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")]

extensions = [
    Extension(
        "wordllama.algorithms.splitter",
        ["wordllama/algorithms/splitter.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "wordllama.algorithms.deduplicate_helpers",
        ["wordllama/algorithms/deduplicate_helpers.pyx"],
        include_dirs=[numpy_include],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "wordllama.algorithms.kmeans",
        ["wordllama/algorithms/kmeans.pyx"],
        include_dirs=[numpy_include],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "wordllama.algorithms.splitter",
        ["wordllama/algorithms/splitter.pyx"],
        include_dirs=[],
        define_macros=[],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "wordllama.algorithms.find_local_minima",
        ["wordllama/algorithms/find_local_minima.pyx"],
        include_dirs=[numpy_include],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "wordllama.algorithms.vector_similarity",
        ["wordllama/algorithms/vector_similarity.pyx"],
        include_dirs=[numpy_include],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="Embedding and lightweight NLP utility.",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        annotate=True,
    ),
    zip_safe=False,
    install_requires=["numpy"],
)
