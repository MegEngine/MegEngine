# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import collections
import textwrap
import os
import hashlib
import struct

class member_defs:
    """contain classes to define members of an opr param"""

    Dtype = collections.namedtuple('Dtype', ['cname', 'pycvt', 'pyfmt',
                                             'cppjson', 'cname_attr'])
    Dtype.__new__.__defaults__ = ('', )
    uint32 = Dtype('uint32_t', 'int', 'I', 'NumberInt')
    uint64 = Dtype('uint64_t', 'int', 'Q', 'NumberInt',
                   'alignas(sizeof(uint64_t)) ')
    int32 = Dtype('int32_t', 'int', 'i', 'NumberInt')
    float32 = Dtype('float', 'float', 'f', 'Number')
    float64 = Dtype('double', 'float', 'd', 'Number')
    dtype = Dtype('DTypeEnum', '_as_dtype_num', 'I', 'Number')
    bool = Dtype('bool', 'bool', '?', 'Bool')

    class Base:
        pass


    class Doc:
        """wrap an identifier to associate document

        note: if the doc starts with a linebreak, it would not be reforamtted.
        """
        __slots__ = ['id', 'doc']

        def __init__(self, id_, doc):
            assert isinstance(id_, str) and isinstance(doc, str), (id_, doc)
            self.id = id_
            self.doc = doc

        @property
        def no_reformat(self):
            """whether reformat is disallowed for this doc string"""
            return self.doc.startswith('\n')

        @property
        def raw_lines(self):
            """the doc lines when ``no_format`` is true"""
            ret = self.doc.split('\n')
            assert not ret[0]
            return ret[1:]

        @classmethod
        def make(cls, v):
            """make doc object from str or doc"""
            if isinstance(v, cls):
                return v
            assert isinstance(v, str)
            return cls(v, '')

        def __str__(self):
            return self.id

        def __eq__(self, rhs):
            if isinstance(rhs, str):
                return self.id == rhs
            return (isinstance(rhs, Doc) and
                    (self.id, self.doc) == (rhs.id, rhs.doc))


    class Enum(Base):
        """define an enum; the result would contain both an enum class def and its
        corresponding data field

        :param default: index of default member value

        :attr name_field: name of the data field of this enum in the param
            struct
        :attr member_alias: list of (member, alias) pairs
        """
        __slots__ = ['name', 'name_field', 'members', 'default',
                     'member_alias']

        all_enums = {}
        """(param_name, name) => enum"""

        def __init__(self, param_name, name, name_field, members, default,
                     member_alias):
            name = member_defs.Doc.make(name)
            assert name.id[0].isupper()
            members = tuple(map(member_defs.Doc.make, members))
            if isinstance(default, str):
                if default not in name_field:
                    raise ValueError(
                        "Default value '{}' does not exist.".format(default))
                default = name_field.index(default)
            assert isinstance(default, int)
            self.name = name
            self.name_field = self.get_name_field(name.id, name_field)
            self.members = members
            self.default = default

            self.all_enums[(param_name, name.id)] = self

            assert isinstance(member_alias, list)
            self.member_alias = member_alias

        @classmethod
        def get_name_field(cls, name, name_field):
            if name_field is None:
                name_field = name[0].lower() + name[1:]
            assert isinstance(name_field, str)
            return name_field

    class Field(Base):
        """define a normal data field"""
        __slots__ = ['name', 'dtype', 'default']

        def __init__(self, name, dtype, default):
            assert isinstance(dtype, member_defs.Dtype)
            self.name = member_defs.Doc.make(name)
            self.dtype = dtype
            self.default = default

    class Const(Base):
        """define a const data field"""
        __slots__ = ['name', 'dtype', 'default']

        def __init__(self, name, dtype, default):
            assert isinstance(dtype, member_defs.Dtype)
            self.name = member_defs.Doc.make(name)
            self.dtype = dtype
            self.default = default

    class EnumAlias(Base):
        """alias of enum type from another param"""
        __slots__ = ['name', 'name_field', 'src_class', 'src_name', 'default']

        def __init__(self, name, name_field, src_class, src_name, default):
            self.name = name
            self.name_field = member_defs.Enum.get_name_field(name, name_field)
            self.src_class = src_class
            if src_name is None:
                src_name = name
            self.src_name = src_name
            self.default = default

        @property
        def src_enum(self):
            """source Enum class"""
            return member_defs.Enum.all_enums[(self.src_class, self.src_name)]

        def get_default(self):
            """get default index; fallback to src index if default is not
            set"""
            if self.default is None:
                return self.src_enum.default
            return self.default


class ParamDef:
    """"""
    __all_tags = set()
    all_param_defs = []

    __slots__ = ['name', 'members', 'tag', 'is_legacy']

    def __init__(self, name, doc='', *, version=0, is_legacy=False):
        self.members = []
        self.all_param_defs.append(self)
        h = hashlib.sha256(name.encode('utf-8'))
        if version:
            h.update(struct.pack('<I', version))
        if is_legacy:
            name += 'V{}'.format(version)
        self.name = member_defs.Doc(name, doc)
        self.tag = int(h.hexdigest()[:8], 16)
        self.is_legacy = is_legacy
        if self.tag < 1024:
            self.tag += 1024
        assert self.tag not in self.__all_tags, (
            'tag hash confliction: name={} tag={}'.format(name, self.tag))
        self.__all_tags.add(self.tag)

    def add_fields(self, dtype, *names_defaults):
        assert isinstance(dtype, str)
        dtype = getattr(member_defs, dtype)
        assert len(names_defaults) % 2 == 0
        for i, j in zip(names_defaults[::2], names_defaults[1::2]):
            self.members.append(member_defs.Field(i, dtype, j))
        return self

    def add_enum(self, name, *members, default=0, name_field=None,
                 member_alias=[]):
        self.members.append(member_defs.Enum(
            self.name.id, name, name_field, members, default, member_alias))
        return self

    def add_enum_alias(self, name, src_class, src_name=None, name_field=None,
                       default=None):
        self.members.append(member_defs.EnumAlias(
            name, name_field, src_class, src_name, default))
        return self

    def add_const(self, dtype, *names_defaults):
        assert isinstance(dtype, str)
        dtype = getattr(member_defs, dtype)
        assert len(names_defaults) % 2 == 0
        for i, j in zip(names_defaults[::2], names_defaults[1::2]):
            self.members.append(member_defs.Const(i, dtype, j))
        return self


class WriterBase:
    """base class for output file writer"""

    _fout = None
    _input_hash = None
    _cur_class = None

    def __call__(self, fout):
        self._fout = fout

    def set_input_hash(self, h):
        self._input_hash = h
        return self

    def _get_header(self):
        return 'generated by {} for {}'.format(
            os.path.basename(__file__),
            self._input_hash
        )

    def _process(self, defs):
        dispatch = {
            member_defs.Enum: self._on_member_enum,
            member_defs.EnumAlias: self._on_member_enum_alias,
            member_defs.Field: self._on_member_field,
            member_defs.Const: self._on_const_field
        }
        for i in defs:
            assert isinstance(i, ParamDef)
            if i.is_legacy:
                continue
            self._cur_class = i.name
            self._on_param_begin(i)
            for j in i.members:
                dispatch[type(j)](j)
            self._on_param_end(i)

    def _on_param_begin(self, p):
        """:type p: :class:`.ParamDef`"""

    def _on_param_end(self, p):
        """:type p: :class:`.ParamDef`"""

    def _on_member_enum(self, e):
        """:type p: :class:`.Enum`"""

    def _on_member_enum_alias(self, e):
        """:type p: :class:`.EnumAlias`"""

    def _on_member_field(self, f):
        """:type p: :class:`.Field`"""

    def _on_const_field(self, f):
        """:type p: :class:`.Const`"""


class IndentWriterBase(WriterBase):
    _cur_indent = ''

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


class PyWriter(IndentWriterBase):

    _static_members = None
    _non_static_members = None
    _enums = None
    _enum_map = None

    def __call__(self, fout, defs):
        super().__call__(fout)
        self._enum_map = {}
        self._write('// %s', self._get_header())
        self._write('#include "megbrain/imperative/opdef/all.h"')
        self._write('')
        self._write('using namespace mgb::imperative;')
        self._write('')
        self._process(defs)

    def _on_param_begin(self, p):
        self._enums = []
        self._non_static_members = []
        self._static_members = []

    def _reg_enum_single(self, cur_def, e):
        alias = None
        if isinstance(e, member_defs.Enum):
            src = e
        else:
            assert isinstance(e, member_defs.EnumAlias)
            src = e.src_enum
            alias = e

        src_py_name = self._enum_map.get(src, None)
        if src_py_name is not None:
            py_name = '{}{}Enum'.format(cur_def, src.name if alias is None else alias.name)
            self._write('m.attr("{}") = m.attr("{}");\n'.format(py_name, src_py_name))
            return

        if alias is None:
            enum_name = str(src.name)
        else:
            enum_name = str(alias.name)
        c_name = 'opdef::{}::{}'.format(cur_def, enum_name)
        py_name = '{}{}Enum'.format(cur_def, enum_name)
        self._write('py::enum_<{}>(m, "{}")'.format(c_name, py_name), indent=1)
        for i in src.members:
            self._write('.value("{0}", {1}::{0})'.format(i, c_name))
        self._write(';\n', indent=-1)
        self._enum_map[src] = py_name

    def _on_param_end(self, p):
        cur_def = '{}Def'.format(p.name)
        for e in self._enums:
            self._reg_enum_single(cur_def, e)
        self._write('py::class_<opdef::{0}>(m, "{0}")'.format(cur_def), indent=1)
        # TODO: use ctor with given default value
        self._write('.def(py::init<>())')
        for i in self._static_members:
            assert isinstance(i, member_defs.Const)
            self._write('.def_property_readonly_static("{0}", []() {{ return opdef::{1}::{0}; }})'.format(i.name, cur_def))
        for i in self._non_static_members:
            fname = None
            if isinstance(i, member_defs.Field):
                fname = i.name
            else:
                assert isinstance(i, (member_defs.Enum, member_defs.EnumAlias))
                fname = i.name_field
            self._write('.def_readwrite("{0}", &opdef::{1}::{0})'.format(fname, cur_def))
        self._write(';\n', indent=-1)


    def _on_member_enum(self, e,):
        self._enums.append(e)
        self._non_static_members.append(e)

    def _on_member_enum_alias(self, e):
        self._enums.append(e)
        self._non_static_members.append(e)

    def _on_member_field(self, f):
        self._non_static_members.append(f)

    def _on_const_field(self, f):
        self._static_members.append(f)


class CPPWriter(IndentWriterBase):
    _param_namespace = 'opdef'

    _ctor_args = None
    """list of (text in func param, var name); func param name must be var name
    appended by an underscore"""
    _non_static_members = None

    def __call__(self, fout, defs):
        super().__call__(fout)
        self._write('// %s', self._get_header())
        self._write('#pragma once')
        self._write('#include "megdnn.h"')
        # which defined in megbrain/tools/param_defs/mgb_opr_param_defs.py
        self._write('#include "megbrain/opr/param_defs.h"')
        self._write('#include <stdint.h>')
        self._write('namespace mgb {')
        self._write('namespace imperative {')
        self._write('namespace %s {', self._param_namespace)
        self._write('namespace {')
        self._write('#include "megdnn/dtype.h"')
        self._write('using DTypeEnum = megdnn::DTypeEnum;')
        self._write('} // anonymous namespace')
        self._process(defs)
        self._write('} // namespace %s', self._param_namespace)
        self._write('} // namespace imperative')
        self._write('} // namespace mgb')
        self._write('// vim: syntax=cpp.doxygen')

    def _on_param_begin(self, p):
        self._write('struct %sDef {', p.name, indent=1)
        self._ctor_args = []
        self._non_static_members = []

    def _add_ctor_args(self, typename, default, varname):
        self._ctor_args.append((
            '{} {}_={}'.format(typename, varname, default),
            varname))

    def _on_param_end(self, p):
        '''
        MegDNN param structures are not packed and we need to initialize the structure
        paddings to zero or it would break MegBrain hash system. We do memset(0) in default
        ctor and use a trick, wrapping non-static members in a anonymous union which would
        copy the object representation in its default copy/move ctor, for copy/move ctor.
        > The implicitly-defined copy/move constructor for a non-union class X performs
        > a memberwise copy/move of its bases and members. [class.copy.ctor 14]
        > The implicitly-defined copy/move constructor for a union X copies the object
        > representation (6.9) of X. [class.copy.ctor 15]
        '''
        if self._non_static_members:
            self._write('union { struct {')
            for i in self._non_static_members:
                if isinstance(i, member_defs.Field):
                    self._write('%s%s %s;', i.dtype.cname_attr, i.dtype.cname, i.name)
                else:
                    assert isinstance(i, (member_defs.Enum, member_defs.EnumAlias))
                    self._write('%s %s;', i.name, i.name_field)
            self._write('}; };')
        param_list = []
        if self._ctor_args:
            pdefs, varnames = zip(*self._ctor_args)
            self._write('%sDef(%s) {', p.name, ', '.join(pdefs), indent=1)
            self._write('memset(this, 0, sizeof(*this));')
            for var in varnames:
                self._write('this->%s = %s_;', var, var)
                param_list.append(str(var))
            self._write('}', indent=-1)
        self._write('megdnn::param::%s param() {', self._cur_class, indent=1)
        self._write('return {%s};', ','.join(param_list))
        self._write('}', indent=-1)
        self._write('};\n', indent=-1)


    def __on_member_enum(self, e, default_value):
        self._write('using %s = megdnn::param::%s::%s;', e.name, self._cur_class, e.name)
        self._non_static_members.append(e)
        self._add_ctor_args(e.name, default_value, e.name_field)

    def _on_member_enum(self, e,):
        self.__on_member_enum(e, '{}::{}'.format(e.name, e.members[e.default]))

    def _on_member_enum_alias(self, e):
        self.__on_member_enum(e, '{}::{}'.format(e.name, e.src_enum.members[e.get_default()]))

    def _on_member_field(self, f):
        self._non_static_members.append(f)
        self._add_ctor_args(f.dtype.cname, f.default, f.name)

    def _on_const_field(self, f):
        if 'int' in f.dtype.cname:
            self._write('static constexpr %s%s %s = %s;', f.dtype.cname_attr, f.dtype.cname, f.name, f.default)
        else:
            self._write('static const %s%s %s = %s;', f.dtype.cname_attr, f.dtype.cname, f.name, f.default)

def main():
    parser = argparse.ArgumentParser(
        'generate opr param defs from description file')
    parser.add_argument('-t', '--type', choices=['c++', 'py'], default='c++',
                        help='output type')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    with open(args.input) as fin:
        inputs = fin.read()
        exec(inputs, {'pdef': ParamDef, 'Doc': member_defs.Doc})
        input_hash = hashlib.sha256()
        input_hash.update(inputs.encode(encoding='UTF-8'))
        input_hash = input_hash.hexdigest()

    if args.type == 'py':
        writer = PyWriter()
    else:
        writer = CPPWriter()

    with open(args.output, 'w') as fout:
        writer.set_input_hash(input_hash)(fout, ParamDef.all_param_defs)

if __name__ == '__main__':
    main()
