# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from io import StringIO
import re
import argparse
import subprocess
import os
import textwrap
import inspect


def camel2underscore(
        name, *,
        first_cap_re=re.compile('([A-Z])([A-Z][a-z]+)'),
        all_cap_re = re.compile('([a-z])([A-Z]+)')):
    if name.isupper():
        return name.lower()
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()


def caller_lineno(level=1):
    f = inspect.stack()[level+1]
    return '%s:%d' % (f.filename, f.lineno)


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


class Context:
    fout = None

    def __init__(self):
        self.fout = StringIO()
        self.indent = 0
        self.generated = []
        self.skipped = []

    def write(self, text, *fmt, indent=0):
        text = textwrap.dedent(text)
        text = textwrap.indent(text, ' '*4*(self.indent + indent))
        text = text % fmt
        if not text.endswith('\n'):
            text += '\n'
        self.fout.write(text)

    def _gen_signature(self, params, *, have_config=True,
                       has_out_dtype=False):
        sig = ['self', '*']

        for i, _ in params:
            sig.append('{}=None'.format(i))

        if have_config:
            sig.extend(['name=None', 'comp_node=None', 'config=None'])
            if has_out_dtype:
                sig.append('dtype=None')

        if params:
            sig.append('**kwargs')

        if sig[-1] == '*':
            sig.pop()
        return ', '.join(sig)

    def _write_canonize_inputs(self, inputs, convert_inputs,
                               convert_inputs_args=None,
                               has_out_dtype=False):
        self._write_gen_config(has_out_dtype)
        inputs = list(map(str, inputs))
        if convert_inputs_args is None:
            if inputs[0][0] == '*':
                arg = inputs[0][1:]
            else:
                arg = '[{}]'.format(', '.join(inputs))
        else:
            arg = convert_inputs_args
        self.write('inputs = helper.%s(%s, config=config)',
                   convert_inputs, arg)

    def _write_gen_config(self, has_out_dtype=False):
        self.write('''\
            config = config or Config()
            if name:
                config.name = name
            if comp_node:
                config.comp_node = comp_node
            ''')
        if has_out_dtype:
            self.write('''\
                if dtype:
                    config.dtype = dtype
                ''')
        self.write('self.config = config')

    def _write_make_params(self, params):
        for pname, ptype in params:
            self.write('self.%s = helper.make_param(%s, param_defs.%s, kwargs)',
                pname, pname, ptype)
        self.write('assert not kwargs, "extra kwargs: {}".format(kwargs)')

    def _write_doc(self, inputs, params, desc):
        self.write('"""')
        if isinstance(desc, Doc):
            assert desc._id is None
            self.write(desc.doc)
        elif desc:
            for i in textwrap.wrap(desc, 75):
                self.write(i)

        self.write('')
        for i in inputs:
            name = str(i)
            typestr = ':class:`.Tensor`'
            if name[0] == '*':
                name = name[1:]
                typestr = 'list of ' + typestr
            if isinstance(i, Doc):
                self.write(':param %s: %s', name, i.doc)
                if i.typestr is not None:
                    typestr = i.typestr
            if typestr:
                if not isinstance(i, Doc):
                    self.write(':param %s: ', name)
                self.write(':type %s: %s', name, typestr)

        for pname, ptype in params:
            self.write(':param %s: ', pname)
            self.write(':type %s: :class:`~megbrain.opr_param_defs.%s`',
                        pname, ptype)

        self.write(':param comp_node: see doc for *config*')
        self.write(':param name: see doc for *config*')
        self.write(
            ':param config: give a :class:`.OperatorNodeConfig` object to set '
            'operator name and comp node. This can also be achieved by passing '
            '*comp_node* and *name* separately.')

        self.write('"""')

    def _write_return(self, name, outputs):
        self.write('opdef = helper.PodOpVisitor("%s", config, params)', name)
        self.write('outputs = helper.create_op(opdef, inputs)')
        if outputs:
            self.write('outputs = [outputs[i] for i in %s]',
                        list(map(int, outputs)))
        self.write('return helper.convert_outputs(outputs)')

    def decl_opr(self, name, *, inputs, params, desc=None, pyname=None,
                 canonize_input_vars=None,
                 canonize_input_vars_args=None, body=None,
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
        if body:
            self.skipped.append(name)
            return

        body = body or []
        if isinstance(params, str):
            params = [('param', params)]
        assert params

        self.write('# %s', caller_lineno())
        self.write('class %s(PodOpVisitor):', name)
        self.indent += 1

        param_names, _ = zip(*params)
        self.write('param_names = (%s,)', ', '.join(map('"{}"'.format, param_names)))
        self.write('name = "%s"', '{}V{}'.format(name, version) if version else name)
        self.write('\n')

        self.write('def __init__(%s):',
                    self._gen_signature(params,
                                        has_out_dtype=has_out_dtype))
        self.indent += 1

        self._write_gen_config(has_out_dtype=has_out_dtype)
        self.write('\n')

        self._write_make_params(params)

        self.write('\n')
        self.indent -= 2

        self.generated.append(name)

    def decl_raw_opr(self, name, *, inputs, inputs_cvt=[], body=None,
                     desc=None, local_defs=[], have_config=True):
        self.skipped.append(name)

    def get_str(self):
        return self.fout.getvalue()

    def all_list(self):
        buf = StringIO()
        print(
            '[',
            *('    "%s",' % i for i in self.generated),
            ']',
            sep='\n',
            file=buf
        )
        return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description='generate operator function def code from decl file')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    gen = Context()
    exec_globals = {
        'decl_opr': gen.decl_opr,
        'decl_raw_opr': gen.decl_raw_opr,
        'Doc': Doc,
        'camel2underscore': camel2underscore,
    }
    for i in args.inputs:
        print('generate ops from {}'.format(i))
        with open(i) as fin:
            exec(compile(fin.read(), i, 'exec'), exec_globals)

    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], universal_newlines=True,
            cwd=os.path.dirname(os.path.realpath(__file__))).strip()
    except:
        git_commit = 'NOT_A_GIT_REPO'

    def relpath(*args):
        d = os.path.dirname(__file__)
        return os.path.join(d, *args)

    with open(relpath('ops.tpl.py')) as fin:
        with open(args.output, 'w') as fout:
            fout.write(fin.read()
                       .replace('{%all%}', gen.all_list())
                       .replace('{%body%}', gen.get_str())
                       .replace('{%git_commit%}', git_commit))

    print('Skipped:')
    print(*gen.skipped, sep='\n')

if __name__ == '__main__':
    main()
