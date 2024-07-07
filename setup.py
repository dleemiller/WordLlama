from setuptools import setup, find_packages

setup(
    name="wordllama",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "safetensors", "tokenizers", "toml"],
    extras_require={
        "train": ["torch", "transformers", "sentence-transformers", "datasets"]
    },
    entry_points={
        "console_scripts": [
            "wordllama=wordllama.wordllama:main",
        ],
    },
    include_package_data=True,
    package_data={
        # Include all toml files in the package
        "": ["*.toml"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
