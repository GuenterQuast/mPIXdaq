# data acquisition, visualisation and analsys for ADVACAM miniPIX EDU
#   contains ADVACAM libraries and miniPIX EDU firmware

from setuptools import setup
import sys
import pathlib

pkg_name = 'mpixdaq'

# The directory containing this file
HERE = pathlib.Path(__file__).parent

sys.path[0] = pkg_name
import _version_info
_version = _version_info._get_version_string()

setup(
    name='mpixdaq',
    version=_version,
    packages=['mpixdaq'],
    package_dir={'mpixdaq': 'mpixdaq'},
    package_data={'mpixdaq': ['advacam*/*.so',
                              'advacam*/*.ini','advacam*/factory/*', 'data/*']},
    scripts=['run_mPIXdaq.py'],
    description='Provides ADVACAM pixet API to python users',
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url='',
    author='Guenter Quast',
    author_email='Guenter.Quast@kit.edu',
    keywords=['pip', 'minipix', 'advacam'],
    classifiers=[
        #"Development Status :: 5 - Production/Stable",
        'Development Status :: 4 - Beta',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        ],
    )
