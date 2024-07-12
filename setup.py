from setuptools import setup, Extension, find_packages

# Setup configuration
setup(
    name="wordllama",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "wordllama=wordllama.wordllama:main",
        ],
    },
    include_package_data=True,
    package_data={
        "wordllama": ["**/*.toml"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

