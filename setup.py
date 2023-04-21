"""TRINIDI package configuration."""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f.readlines()]


setup(
    name="TRINIDI",
    version="0.0.0",
    description="Time-of-Flight Resonance Imaging with Neutrons for Isotopic Density Inference (TRINIDI)",
    author="Thilo Balke",
    author_email="thilo.balke@gmail.com",
    url="https://github.com/lanl/trinidi",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
)
