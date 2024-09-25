# setup.py
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pla_net',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[requirements],
    classifiers=[],
    python_requires='>=3.8',
)