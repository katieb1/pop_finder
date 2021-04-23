# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['pop_finder']

package_data = \
{'': ['*']}

install_requires = \
['h5py==2.10.0',
 'keras-tuner==1.0.2',
 'matplotlib==3.3.2',
 'numpy==1.19.5',
 'pandas==1.1.5',
 'scikit-allel==1.3.0',
 'scikit-learn==0.23',
 'scipy>=1.6.0,<2.0.0',
 'seaborn>=0.11.1,<0.12.0',
 'tensorflow-cpu==2.4.1',
 'tqdm>=4.59.0,<5.0.0',
 'wheel>=0.35.1,<0.36.0',
 'zarr>=2.6.1,<3.0.0']

extras_require = \
{'tf': ['tensorflow==2.4.1']}

entry_points = \
{'console_scripts': ['pop_finder_classifier = pop_finder.cli_classifier:main',
                     'pop_finder_regressor = pop_finder.cli_regressor:main']}

setup_kwargs = {
    'name': 'pop-finder',
    'version': '1.0.10',
    'description': '"Python package that uses neural networks for population assignment"',
    'long_description': None,
    'author': 'Katie Birchard',
    'author_email': 'birchardkatie@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'entry_points': entry_points,
    'python_requires': '>=3.7.1,<3.10',
}


setup(**setup_kwargs)

