import os
from setuptools import setup, find_packages

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="shign",
    version="1.0",
    description='Shift two recordings of the same audio event to align them in time',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Bram Kooiman',
    author_email='bramkooiman94@gmail.com',
    packages=['shign'],
    install_requires=[
        'numpy',
        'scipy',
        'soundfile',
        'librosa',
    ],
    include_package_data=True
)
