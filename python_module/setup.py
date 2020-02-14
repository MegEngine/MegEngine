# -*- coding: utf-8 -*-
# This file is part of MegBrain.
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.

import os
import re
import pathlib
from distutils.file_util import copy_file
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext

class PrecompiledExtesion(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class build_ext(_build_ext):

    def build_extension(self, ext):
        if not isinstance(ext, PrecompiledExtesion):
            return super().build_extension(ext)

        if not self.inplace:
            fullpath = self.get_ext_fullpath(ext.name)
            extdir = pathlib.Path(fullpath)
            extdir.parent.mkdir(parents=True, exist_ok=True)

            modpath = self.get_ext_fullname(ext.name).split('.')
            modpath[-1] += '.so'
            modpath = str(pathlib.Path(*modpath).resolve())

            copy_file(modpath, fullpath, verbose=self.verbose, dry_run=self.dry_run)

package_name = 'MegEngine'

v = {}
with open("megengine/version.py") as fp:
    exec(fp.read(), v)
__version__ = v['__version__']

email = 'megengine@megvii.com'
local_version = os.environ.get('LOCAL_VERSION')
if local_version:
    __version__ = '{}+{}'.format(__version__, local_version)

packages = find_packages(exclude=['test'])
package_data = [
    str(f.relative_to('megengine'))
    for f in pathlib.Path('megengine', '_internal', 'include').glob('**/*')
]
package_data += [
    str(f.relative_to('megengine'))
    for f in pathlib.Path('megengine', '_internal', 'lib').glob('**/*')
]
package_data += [
    os.path.join('module', 'pytorch', 'torch_mem_fwd.cpp')
]

setup_kwargs = dict(
    name=package_name,
    version=__version__,
    description='Framework for numerical evaluation with '
    'auto-differentiation',
    author='Megvii Engine Team',
    author_email=email,
    packages=packages,
    package_data={
        'megengine': package_data,
    },
    ext_modules=[PrecompiledExtesion('megengine._internal._mgb')],
    install_requires=[
        'numpy>=1.17',
        'opencv-python',
        'pyarrow',
        'requests',
        'tabulate',
        'tqdm',
    ],
    extras_require={
        'dev': [
            'black==19.10b0',
            'isort==4.3.21',
            'pylint==2.4.3',
            'mypy==0.750',
            'pytest==5.3.0',
            'pytest-sphinx==0.2.2',
        ],
        'data': [
            'scipy',
        ],
    },
    cmdclass={'build_ext': build_ext},
)


setup_kwargs.update(dict(
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: C++',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='megengine deep learning',
    data_files = [("", [
        "../LICENSE",
        "../ACKNOWLEDGMENTS",
    ])]
))

setup(**setup_kwargs)
