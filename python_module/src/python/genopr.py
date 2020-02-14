#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of MegBrain.
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.

from io import StringIO
import re
import argparse
import subprocess
import os
import textwrap

def camel2underscore(
        name, *,
        first_cap_re=re.compile('([A-Z])([A-Z][a-z]+)'),
        all_cap_re = re.compile('([a-z])([A-Z]+)')):
    if name.isupper():
        return name.lower()
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()

class Doc:
    """wrap an identifier and doc"""
    _id = None

    def __init__(self, id_, doc, typestr=None, default=None):
        self._id = id_
        self.doc = doc
        self.typestr = typestr
        self.default = default

    def __str__(self):
        return self._id


class OprGenerator:
    _fout = None
    _cur_indent = ''

    def __init__(self):
        self._fout = StringIO()

    def _indent(self):
        self._cur_indent += ' ' * 4

    def _unindent(self):
        self._cur_indent = self._cur_indent[:-4]

    def _write(self, content, *fmt, indent=0):
        if indent < 0:
            self._unindent()

        self._fout.write(self._cur_indent)
        if fmt:
            content = content % fmt
        self._fout.write(content)
        self._fout.write('\n')

        if indent > 0:
            self._indent()

    def _gen_signature(self, inputs, params, *, have_config=True,
                       has_out_dtype=False):
        assert inputs
        sig = []
        for inp in inputs:
            name = str(inp)
            if name.startswith('*'):
                assert name[1:].isidentifier()
                assert inp is inputs[-1]
            else:
                assert name.isidentifier()
            if isinstance(inp, Doc) and inp.default is not None:
                name += '={}'.format(inp.default)
            sig.append(name)
        if not str(inputs[-1]).startswith('*'):
            sig.append('*')
        for i, _ in params:
            sig.append('{}=None'.format(i))

        if have_config:
            sig.extend(['name=None', 'comp_node=None', 'config=None'])
            if 'comp_graph' not in map(str, inputs):
                sig.append('comp_graph=None')
            if has_out_dtype:
                assert 'dtype' not in map(str, inputs)
                sig.append('dtype=None')

        if params:
            sig.append('**kwargs')

        if sig[-1] == '*':
            sig.pop()
        return ', '.join(sig)

    def _write_canonize_inputs(self, inputs, canonize_input_vars,
                               canonize_input_vars_args=None,
                               has_out_dtype=False):
        self._write_gen_config(has_out_dtype)
        inputs = list(map(str, inputs))
        if canonize_input_vars_args is None:
            if inputs[0][0] == '*':
                arg = inputs[0][1:]
            else:
                arg = '[{}]'.format(', '.join(inputs))
        else:
            arg = canonize_input_vars_args
        self._write('all_inputs = _helper.%s(%s, '
                    'comp_graph=comp_graph, config=config)',
                    canonize_input_vars, arg)

    def _write_gen_config(self, has_out_dtype=False):
        if not has_out_dtype:
            self._write('config = _helper.gen_config(name, comp_node, config)')
        else:
            self._write('config = _helper.gen_config(name, comp_node, config, dtype)')

    def _write_make_params(self, params, body):
        for pname, ptype in params:
            self._write('%s = _helper.cvt_to_opr_param_def(%s, '
                        '_opr_param_defs.%s, kwargs)',
                        pname, pname, ptype)
        self._write('assert not kwargs, "extra kwargs: {}".format(kwargs)')

        for i in body:
            self._write(i)

        self._write('all_params = []')
        for pname, _ in params:
            self._write('all_params.append(%s.serialize())',
                        pname)

    def _write_doc(self, inputs, params, desc):
        self._write('"""')
        if isinstance(desc, Doc):
            assert desc._id is None
            self._write(desc.doc)
        elif desc:
            for i in textwrap.wrap(desc, 75):
                self._write(i)

        self._write('')
        for i in inputs:
            name = str(i)
            typestr = ':class:`.SymbolVar`'
            if name[0] == '*':
                name = name[1:]
                typestr = 'list of ' + typestr
            if isinstance(i, Doc):
                self._write(':param %s: %s', name, i.doc)
                if i.typestr is not None:
                    typestr = i.typestr
            if typestr:
                if not isinstance(i, Doc):
                    self._write(':param %s: ', name)
                self._write(':type %s: %s', name, typestr)

        for pname, ptype in params:
            self._write(':param %s: ', pname)
            self._write(':type %s: :class:`~megbrain.opr_param_defs.%s`',
                        pname, ptype)

        self._write(':param comp_node: see doc for *config*')
        self._write(':param name: see doc for *config*')
        self._write(
            ':param config: give a :class:`.OperatorNodeConfig` object to set '
            'operator name and comp node. This can also be achieved by passing '
            '*comp_node* and *name* separately.')

        if 'comp_graph' not in map(str, inputs):
            self._write(
                ':param comp_graph: If all inputs are immediate numbers, '
                '*comp_graph* and *comp_node* must be provided '
                'so the input values can be put on appropriate '
                'computing graph and computing node')
        self._write('"""')

    def _write_return(self, name, outputs):
        self._write('outputs = _mgb._create_opr('
                    '"%s", all_inputs, all_params, config)', name)
        if outputs:
            self._write('outputs = [outputs[i] for i in %s]',
                        list(map(int, outputs)))
        self._write('return _helper.cvt_opr_result(outputs, '
                    '**cvt_result_kwargs)')

    def decl_opr(self, name, *, inputs, params, desc=None, pyname=None,
                 canonize_input_vars='canonize_input_vars',
                 canonize_input_vars_args=None, body=[],
                 outputs=None, version=0, has_out_dtype=False):
        """
        :param inputs: name of variable inputs; a name starting with `*' means
            a list of vars
        :type inputs: list of str
        :param params: (param name, param type) pairs; it can be a single
            string representing the param type, and param name defaults to
            'param'
        :type params: list of pair of str, or str
        :param pyname: python function name
        :param body: extra statements to be placed before calling _create_opr
        :param outputs: the indices of output vars to be selected from raw opr
            result
        """
        if isinstance(params, str):
            params = [('param', params)]
        assert params

        if pyname is None:
            pyname = camel2underscore(name)

        self._write('def %s(%s):', pyname,
                    self._gen_signature(inputs, params,
                                        has_out_dtype=has_out_dtype), indent=1)
        self._write_doc(inputs, params, desc)
        self._write_canonize_inputs(
            inputs, canonize_input_vars,
            canonize_input_vars_args=canonize_input_vars_args,
            has_out_dtype=has_out_dtype)
        self._write('cvt_result_kwargs = {}')
        self._write_make_params(params, body)
        if version:
            name += 'V{}'.format(version)
        self._write_return(name, outputs)
        self._write('', indent=-1)

    def decl_raw_opr(self, name, *, inputs, inputs_cvt=[], body=None,
                     desc=None, local_defs=[], have_config=True):
        """declare a raw operator that is forwarded to _mgb._Opr; if *body* is
        given, a custom implemented can be provided

        :param inputs_cvt: list of (input name, cvt) pairs, where cvt is name
            of the function to convert that input
        :param body: list of statements to produce output, or None
        :param local_defs: list of statements to be prepended before generated
            code
        """
        self._write('def %s(%s):', name,
                    self._gen_signature(inputs, [], have_config=have_config),
                    indent=1)
        self._write_doc(inputs, [], desc)
        if have_config:
            self._write_gen_config()
        for i in local_defs:
            self._write(i)
        for k, v in inputs_cvt:
            self._write('%s = %s(%s)', k, v, k)
        self._write('cvt_result_kwargs = {}')
        if body is None:
            self._write('output = _mgb._Opr.%s(%s, config)',
                        name, ', '.join(map(str, inputs)))
        else:
            for i in body:
                self._write(i)
        self._write(
            'return _helper.cvt_opr_result(output, **cvt_result_kwargs)')
        self._write('', indent=-1)

    def get_str(self):
        return self._fout.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description='generate operator function def code from decl file')
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()

    gen = OprGenerator()
    exec_globals = {
        'decl_opr': gen.decl_opr,
        'decl_raw_opr': gen.decl_raw_opr,
        'Doc': Doc,
        'camel2underscore': camel2underscore,
    }
    for i in args.inputs:
        print('generate oprs from {}'.format(i))
        with open(i) as fin:
            exec(fin.read(), exec_globals)

    git_commit = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                               stdout=subprocess.PIPE).communicate()[0].strip()
    git_commit = git_commit.decode('utf-8')

    file_rela = lambda *paths: os.path.join(os.path.dirname(__file__), *paths)
    outfile = lambda name: file_rela('../../megengine/_internal', name)
    with open(file_rela('opr_template.py')) as fin:
        with open(outfile('opr.py'), 'w') as fout:
            fout.write(fin.read().
                       replace('{%body%}', gen.get_str()).
                       replace('{%git_commit%}', git_commit))

if __name__ == '__main__':
    main()
