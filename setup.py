#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages


setup(name='dlfm_makam_recognition',
      version='1.0.0',
      author='Sertan Senturk',
      author_email='contact AT sertansenturk DOT com',
      license='agpl 3.0',
      description='Repository of the Makam Recognition Experiments for DLfM '
                  '2016',
      url='http://sertansenturk.com',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          "scikit-learn>=0.17",
          "numpy>=1.9.0",
          "ipyparallel>5.0.0"
      ],
      )
