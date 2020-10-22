# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
import re
import pathlib
import platform
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
            if platform.system() == 'Windows':
                modpath[-1] += '.pyd'
            else:
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
megengine_data = [
    str(f.relative_to('megengine'))
    for f in pathlib.Path('megengine', 'core', 'include').glob('**/*')
]

megengine_data += [
    str(f.relative_to('megengine'))
    for f in pathlib.Path('megengine', 'core', 'lib').glob('**/*')
]


with open('requires.txt') as f:
    requires = f.read().splitlines()
with open('requires-style.txt') as f:
    requires_style = f.read().splitlines()
with open('requires-test.txt') as f:
    requires_test = f.read().splitlines()

prebuild_modules=[PrecompiledExtesion('megengine.core._imperative_rt')]
setup_kwargs = dict(
    name=package_name,
    version=__version__,
    description='Framework for numerical evaluation with '
    'auto-differentiation',
    author='Megvii Engine Team',
    author_email=email,
    packages=packages,
    package_data={
        'megengine': megengine_data,
    },
    ext_modules=prebuild_modules,
    install_requires=requires,
    extras_require={
        'dev': requires_style + requires_test,
        'ci': requires_test,
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
    data_files = [("megengine", [
        "../LICENSE",
        "../ACKNOWLEDGMENTS",
    ])]
))

setup(**setup_kwargs)
