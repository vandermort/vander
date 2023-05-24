import os
from glob import glob
from itertools import chain
from setuptools import setup


def collect(*patterns):
    return set(chain(*(filter(os.path.isfile, glob(p))
                     for p in patterns)))


setup(
      name='vndr',
      packages=['vndr'],
      version='0.0.1',
      author='REDACTED',
      author_email='REDACTED',
      description='Verifiable multi-label classification',
      license='BSD',
      keywords=['classification', 'multilabel', 'nlp', 'machine learning', 'verification'],
      classifiers=[],
      scripts=collect(
        'bin/*',
        'bin/process/*',
        'bin/visualise/*'
      ),
      install_requires=[],
      tests_require=['pytest', 'pytest-sugar', 'pytest-xdist']
      )
