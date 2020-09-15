#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import sys
import re

if sys.version_info[0] != 3 or sys.version_info[1] < 5:
    print('This script requires Python version 3.5')
    sys.exit(1)

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

MIDOUT_TRACE_MAGIC = 'midout_trace v1\n'

class HeaderGen:
    _dtypes = None
    _oprs = None
    _fout = None
    _elemwise_modes = None
    _has_netinfo = False
    _midout_files = None

    _file_without_hash = False

    def __init__(self):
        self._dtypes = set()
        self._oprs = set()
        self._elemwise_modes = set()
        self._graph_hashes = set()
        self._midout_files = []

    _megvii3_root_cache = None
    @classmethod
    def get_megvii3_root(cls):
        if cls._megvii3_root_cache is not None:
            return cls._megvii3_root_cache
        wd = Path(__file__).resolve().parent
        while wd.parent != wd:
           workspace_file = wd / 'WORKSPACE'
           if workspace_file.is_file():
               cls._megvii3_root_cache = str(wd)
               return cls._megvii3_root_cache
           wd = wd.parent
        raise RuntimeError('This script is supposed to run in megvii3.')

    def extend_netinfo(self, data):
        self._has_netinfo = True
        if 'hash' not in data:
            self._file_without_hash = True
        else:
            self._graph_hashes.add(str(data['hash']))
        for i in data['dtypes']:
            self._dtypes.add(i)
        for i in data['opr_types']:
            self._oprs.add(i)
        for i in data['elemwise_modes']:
            self._elemwise_modes.add(i)

    def extend_midout(self, fname):
        self._midout_files.append(fname)

    def generate(self, fout):
        self._fout = fout
        self._write_def('MGB_BINREDUCE_VERSION', '20190219')
        if self._has_netinfo:
            self._write_dtype()
            self._write_elemwise_modes()
            self._write_oprs()
            self._write_hash()
        self._write_midout()
        del self._fout

    def strip_opr_name_with_version(self, name):
        pos = len(name)
        t = re.search(r'V\d+$', name)
        if t:
            pos = t.start()
        return name[:pos]

    def _write_oprs(self):
        defs = ['}',  'namespace opr {']
        already_declare = set()
        already_instance = set()
        for i in self._oprs:
            i = self.strip_opr_name_with_version(i)
            if i in already_declare:
                continue
            else:
                already_declare.add(i)

            defs.append('class {};'.format(i))
        defs.append('}')
        defs.append('namespace serialization {')
        defs.append("""
            template<class Opr, class Callee>
            struct OprRegistryCaller {
            }; """)
        for i in sorted(self._oprs):
            i = self.strip_opr_name_with_version(i)
            if i in already_instance:
                continue
            else:
                already_instance.add(i)

            defs.append("""
                template<class Callee>
                struct OprRegistryCaller<opr::{}, Callee>: public
                    OprRegistryCallerDefaultImpl<Callee> {{
                }}; """.format(i))
        self._write_def('MGB_OPR_REGISTRY_CALLER_SPECIALIZE', defs)

    def _write_elemwise_modes(self):
        with tempfile.NamedTemporaryFile() as ftmp:
            fpath = os.path.realpath(ftmp.name)
            subprocess.check_call(
                ['./brain/megbrain/dnn/scripts/gen_param_defs.py',
                 '--write-enum-items', 'Elemwise:Mode',
                 './brain/megbrain/dnn/scripts/opr_param_defs.py',
                 fpath],
                cwd=self.get_megvii3_root()
            )

            with open(fpath) as fin:
                mode_list = [i.strip() for i in fin]

        for i in mode_list:
            if i in self._elemwise_modes:
                content = '_cb({})'.format(i)
            else:
                content = ''
            self._write_def(
                '_MEGDNN_ELEMWISE_MODE_ENABLE_IMPL_{}(_cb)'.format(i), content)
        self._write_def('MEGDNN_ELEMWISE_MODE_ENABLE(_mode, _cb)',
                        '_MEGDNN_ELEMWISE_MODE_ENABLE_IMPL_##_mode(_cb)')

    def _write_dtype(self):
        if 'Float16' not in self._dtypes:
            # MegBrain/MegDNN used MEGDNN_DISABLE_FLOT16 to turn off float16
            # support in the past; however `FLOT16' is really a typo. We plan to
            # change MEGDNN_DISABLE_FLOT16 to MEGDNN_DISABLE_FLOAT16 soon.
            # To prevent issues in the transition, we decide to define both
            # macros (`FLOT16' and `FLOAT16') here.
            #
            # In the future when the situation is settled and no one would ever
            # use legacy MegBrain/MegDNN, the `FLOT16' macro definition can be
            # safely deleted.
            self._write_def('MEGDNN_DISABLE_FLOT16', 1)
            self._write_def('MEGDNN_DISABLE_FLOAT16', 1)

    def _write_hash(self):
        if self._file_without_hash:
            print('WARNING: network info has no graph hash. Using json file '
                  'generated by MegBrain >= 7.28.0 is recommended')
        else:
            defs = 'ULL,'.join(self._graph_hashes) + 'ULL'
            self._write_def('MGB_BINREDUCE_GRAPH_HASHES', defs)

    def _write_def(self, name, val):
        if isinstance(val, list):
            val = '\n'.join(val)
        val = str(val).strip().replace('\n', ' \\\n')
        self._fout.write('#define {} {}\n'.format(name, val))

    def _write_midout(self):
        if not self._midout_files:
            return

        gen = os.path.join(self.get_megvii3_root(), 'brain', 'midout',
                           'gen_header.py')
        cvt = subprocess.run(
            [gen] + self._midout_files,
            stdout=subprocess.PIPE, check=True,
        ).stdout.decode('utf-8')
        self._fout.write('// midout \n')
        self._fout.write(cvt)

def main():
    parser = argparse.ArgumentParser(
        description='generate header file for reducing binary size by '
        'stripping unused oprs in a particular network; output file would '
        'be written to bin_reduce.h',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'inputs', nargs='+',
        help='input files that describe specific traits of the network; '
        'can be one of the following:'
        '  1. json files generated by '
        'megbrain.serialize_comp_graph_to_file() in python; '
        '  2. trace files generated by midout library')
    parser.add_argument('-o', '--output', help='output file',
                        default=os.path.join(HeaderGen.get_megvii3_root(),
                                             'utils', 'bin_reduce.h'))
    args = parser.parse_args()

    gen = HeaderGen()
    for i in args.inputs:
        print('==== processing {}'.format(i))
        with open(i) as fin:
            if fin.read(len(MIDOUT_TRACE_MAGIC)) == MIDOUT_TRACE_MAGIC:
                gen.extend_midout(i)
            else:
                fin.seek(0)
                gen.extend_netinfo(json.loads(fin.read()))

    with open(args.output, 'w') as fout:
        gen.generate(fout)

if __name__ == '__main__':
    main()
