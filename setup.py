"""TRINIDI package configuration."""

import os
from ast import parse

from setuptools import find_packages, setup

name = "trinidi"

with open(os.path.join(name, "__init__.py")) as f:
    version = parse(next(filter(lambda line: line.startswith("__version__"), f))).body[0].value.s

with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f.readlines()]


description = (
    "Time-of-Flight Resonance Imaging with Neutrons for Isotopic Density Inference (TRINIDI)"
)
long_description = """
Time-of-Flight Resonance Imaging with Neutrons for Isotopic Density Inference (TRINIDI) is a Python package for estimating isotopic densities using neutron transmission resonance images.
"""

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    author="Thilo Balke",
    author_email="thilo.balke@gmail.com",
    url="https://github.com/lanl/trinidi",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    license="BSD",
    package_data={"trinidi": ["data/*.npy"]},
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
