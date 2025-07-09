from setuptools import setup

# save ADVACAM libraries and miniPIX EDU firmware
setup(
    name='advacam',
    packages=['advacam'],
    package_dir={'advacam': 'advacam'},
    package_data={'advacam': ['*.so', '*.ini','factory/*']},
    description='Provides ADVACAM pixet API to python users',
    version='1.8.3',
    url='https://yo.com',
    author='Guenter Quast',
    author_email='',
    keywords=['pip', 'advacam']
    )

