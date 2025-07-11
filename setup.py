from setuptools import setup
import sys

pkg_name = 'mpixdaq'
# data acquisition, visualisation and analsys for ADVACAM miniPIX EDU
#   contains ADVACAM libraries and miniPIX EDU firmware

sys.path[0] = pkg_name
import _version_info
_version = _version_info._get_version_string()

setup(
    name='mpixdaq',
    packages=['mpixdaq'],
    package_dir={'mpixdaq': 'mpixdaq'},
    package_data={'mpixdaq': ['*.so', '*.ini','factory/*']},
    description='Provides ADVACAM pixet API to python users',
    version=_version,
    url='',
    author='Guenter Quast',
    author_email='Guenter.Quast@kit.edu',
    keywords=['pip', 'minipix', 'advacam']
    )

