# -*- coding: utf-8 -*-

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
# https://www.python.org/dev/peps/pep-0440
# Public version identifiers: [N!]N(.N)*[{a|b|rc}N][.postN][.devN]
# Local version identifiers: <public version identifier>[+<local version label>]
# PUBLIC_VERSION_POSTFIX use to handle rc or dev info
public_version_postfix = os.environ.get('PUBLIC_VERSION_POSTFIX')
if public_version_postfix:
    __version__ = '{}{}'.format(__version__, public_version_postfix)

local_version = []
strip_sdk_info = os.environ.get('STRIP_SDK_INFO', 'False').lower()
sdk_name = os.environ.get('SDK_NAME', 'cpu')
if 'true' == strip_sdk_info:
    print('wheel version strip sdk info')
else:
    local_version.append(sdk_name)
local_postfix = os.environ.get('LOCAL_VERSION')
if local_postfix:
    local_version.append(local_postfix)
if len(local_version):
    __version__ = '{}+{}'.format(__version__, '.'.join(local_version))

packages = find_packages(exclude=['test'])
megengine_data = [
    str(f.relative_to('megengine'))
    for f in pathlib.Path('megengine', 'core', 'include').glob('**/*')
]

megengine_data += [
    str(f.relative_to('megengine'))
    for f in pathlib.Path('megengine', 'core', 'lib').glob('**/*')
]

megenginelite_data = [
    str(f.relative_to('megenginelite'))
    for f in pathlib.Path('megenginelite').glob('**/*')
]

if platform.system() == 'Windows':
    megenginelite_data.remove('libs\\liblite_shared_whl.pyd')
else:
    megenginelite_data.remove('libs/liblite_shared_whl.so')

sdkname2requres = {'cu118': ['nvidia-cuda-runtime-cu11==11.8.89',
                             'nvidia-cuda-nvrtc-cu11==11.8.89',
                             'nvidia-cudnn-cu11==8.6.0.163',
                             'nvidia-cublas-cu11==11.10.3.66'],
                   }

with open('requires.txt') as f:
    requires = f.read().splitlines()
if os.environ.get("BUILD_WITH_LIBRARY", "false") == "false": 
    if sdk_name in sdkname2requres.keys():
        requires = requires + sdkname2requres[sdk_name]
    
with open('requires-style.txt') as f:
    requires_style = f.read().splitlines()
with open('requires-test.txt') as f:
    requires_test = f.read().splitlines()

prebuild_modules=[PrecompiledExtesion('megengine.core._imperative_rt')]
prebuild_modules.append(PrecompiledExtesion('megenginelite.libs.liblite_shared_whl'))
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
        'megenginelite': megenginelite_data,
    },
    ext_modules=prebuild_modules,
    install_requires=requires,
    extras_require={
        'dev': requires_style + requires_test,
        'ci': requires_test,
    },
    cmdclass={'build_ext': build_ext},
    scripts = ['./megengine/tools/mge'],
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
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
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
