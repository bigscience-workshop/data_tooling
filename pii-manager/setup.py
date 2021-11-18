"""
Create pii-manager as a Python package
"""

import io
import sys

from setuptools import setup, find_packages

from typing import Dict, List

from src.pii_manager import VERSION

PKGNAME = 'pii-manager'
GITHUB_URL = ''
DESC = '''
Process PII fragments contained in text, for different languages & countries
'''

# --------------------------------------------------------------------

def requirements(filename='requirements.txt'):
    '''Read the requirements file'''
    with io.open(filename, 'r') as f:
        return [line.strip() for line in f if line and line[0] != '#']


# --------------------------------------------------------------------

PYTHON_VERSION = (3, 8)

if sys.version_info < PYTHON_VERSION:
    sys.exit('**** Sorry, {} {} needs at least Python {}'.format(
        PKGNAME, VERSION, '.'.join(map(str, PYTHON_VERSION))))


# --------------------------------------------------------------------


setup_args = dict(
    # Metadata
    name=PKGNAME,
    version=VERSION,
    description='Text Anonymization of PII',
    long_description=DESC,
    license='Apache',
    url=GITHUB_URL,
    download_url=GITHUB_URL + '/tarball/v' + VERSION,

    # Locate packages
    packages=find_packages('src'),  # [ PKGNAME ],
    package_dir={'': 'src'},

    # Requirements
    python_requires='>=3.6',
    #install_requires=requirements(),

    # Optional requirements
    extras_require={
        'test': ['pytest', 'nose', 'coverage'],
    },

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    entry_points={'console_scripts': [
        'pii-manage = pii_manager.app.manage:main'
    ]},

    include_package_data=False,
    package_data={
    },

    # Post-install hooks
    cmdclass={},

    keywords=['AURA', '4th Platform', 'Cognitive Computing'],
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'License :: Other/Proprietary License',
        'Development Status :: 4 - Beta',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
    ],
)

if __name__ == '__main__':
    # Add requirements
    setup_args['install_requires'] = requirements()
    # Setup
    setup(**setup_args)
