#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = '0.1.8'

LICENSE='Apache 2.0'

INSTALL_REQUIRES = [
    'cycler==0.10.0',
    'decorator==4.0.11',
    'matplotlib==2.0.0',
    'networkx==1.11',
    'numpy==1.14.0',
    'pandas==0.21.1',
    'scikit-learn==0.18.1',
    'scipy==0.19.0',
    'six==1.10.0',
    'wheel==0.29.0',
    'fire',
    'BlackBoxAuditing>=0.1.26',
    'ggplot'
]

PACKAGES = find_packages()
PACKAGE_DATA = {
  'fairness.data.raw' : ['*.csv','*.txt'],
  'fairness.data.preprocessed' : ['*.csv'],
  'fairness.algorithms.kamishima' : ['kamfadm-2012ecmlpkdd/*', 'kamfadm-2012ecmlpkdd/*/*',
                                     'kamfadm-2012ecmlpkdd/fadm/__init__.py',
                                     'kamfadm-2012ecmlpkdd/*/*/*'],
  'fairness.algorithms.zafar' : ['fair-classification-master/*',
                                 'fair-classification-master/disparate_impact/*',
                                 'fair-classification-master/disparate_impact/run-classifier/*']
}
INCLUDE_PACKAGE_DATA = True
PACKAGE_DIR = {
  'fairness.data' : 'fairness/data',
  'fairness.algorithms.kamishima' : 'fairness/algorithms/kamishima',
  'fairness.algorithms.zafar' : 'fairness/algorithms/zafar'
}

ENTRY_POINTS = {
  'console_scripts': [
      'fairness-benchmark = fairness.benchmark:main',
      'fairness-preprocess = fairness.preprocess:main',
      'fairness-analysis = fairness.analysis:main'
  ],
}

with open("README.md", "r") as fh:
    long_description = fh.read()

    setup(
    name="fairness",
    version=VERSION,
    author="See Authors.txt",
    author_email="fairness@haverford.edu",
    license=LICENSE,
    description="Fairness-aware machine learning: algorithms, comparisons, benchmarking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/algofairness/fairness-comparison",
    packages = PACKAGES,
    package_data = PACKAGE_DATA,
    include_package_data = INCLUDE_PACKAGE_DATA,
    package_dir = PACKAGE_DIR,
    classifiers=(
                 "Programming Language :: Python :: 3",
                 "License :: OSI Approved :: Apache Software License",
                 "Operating System :: OS Independent",
                ),
    install_requires=INSTALL_REQUIRES
    )

