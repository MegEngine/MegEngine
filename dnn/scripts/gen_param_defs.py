#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    FieldDef = collections.namedtuple(
        'FieldDef', ['name', 'cvt', 'fmt', 'default', 'type', 'doc'])
    # see _on_param_end() for the use of those fields

    _cur_param_name = None
    _cur_fields = None
    _cur_struct_fmt = None

    _enum_member2num = None

    def __init__(self, for_imperative=False):
        self._imperative = for_imperative

    def __call__(self, fout, defs):
        super().__call__(fout)
        self._enum_member2num = []
        self._write('# %s', self._get_header())
        self._write('import struct')
        self._write('from . import enum36 as enum')
        self._write(
            'class _ParamDefBase:\n'
            '   def serialize(self):\n'
            '       tag = struct.pack("I", type(self).TAG)\n'
            '       pdata = [getattr(self, i) for i in self.__slots__]\n'
            '       for idx, v in enumerate(pdata):\n'
            '           if isinstance(v, _EnumBase):\n'
            '               pdata[idx] = _enum_member2num[id(v)]\n'
            '       return tag + self._packer.pack(*pdata)\n'
            '\n'
        )
        self._write(
            'class _EnumBase(enum.Enum):\n'
            '   @classmethod\n'
            '   def __normalize(cls, val):\n'
            '       if isinstance(val, str):\n'
            '           if not hasattr(cls, "__member_upper_dict__"):\n'
            '               cls.__member_upper_dict__ = {k.upper(): v\n'
            '                   for k, v in cls.__members__.items()}\n'
            '           val = cls.__member_upper_dict__.get(val.upper(),val)\n'
            '       return val\n'
            '   @classmethod\n'
            '   def convert(cls, val):\n'
            '       val = cls.__normalize(val)\n'
            '       if isinstance(val, cls):\n'
            '           return val\n'
            '       return cls(val)\n'
            '   @classmethod\n'
            '   def _missing_(cls, value):\n'
            '       vnorm = cls.__normalize(value)\n'
            '       if vnorm is not value:\n'
            '           return cls(vnorm)\n'
            '       return super()._missing_(value)\n'
            '\n'
        )
        if not self._imperative:
            self._write(
                'def _as_dtype_num(dtype):\n'
                '    import megbrain.mgb as m\n'
                '    return m._get_dtype_num(dtype)\n'
                '\n'
            )

            self._write(
                'def _as_serialized_dtype(dtype):\n'
                '    import megbrain.mgb as m\n'
                '    return m._get_serialized_dtype(dtype)\n'
                '\n'
            )
        else:
            self._write(
                'def _as_dtype_num(dtype):\n'
                '    import megengine.core._imperative_rt.utils as m\n'
                '    return m._get_dtype_num(dtype)\n'
                '\n'
            )

            self._write(
                'def _as_serialized_dtype(dtype):\n'
                '    import megengine.core._imperative_rt.utils as m\n'
                '    return m._get_serialized_dtype(dtype)\n'
                '\n'
            )

        self._process(defs)
        self._write(
            '''
class SerializedDType(_ParamDefBase):
    TAG = FakeSerializedDType.TAG
    __slots__ = ['dtype']
    class IdentityPacker:
        def pack(self, *args):
            assert all([isinstance(x, bytes) for x in args])
            return b''.join(args)
    _packer = IdentityPacker()
    def __init__(self, dtype):
        """
        :type dtype: :class:`np.dtype` compatible
        """
        self.dtype = _as_serialized_dtype(dtype)
'''
        )
        self._write('_enum_member2num = {\n  %s}',
                    ',\n  '.join(self._enum_member2num))

    def _write_doc(self, doc):
        assert isinstance(doc, member_defs.Doc)
        if not doc.doc:
            return
        if doc.no_reformat:
            self._write('"""')
            for i in doc.raw_lines:
                self._write(i)
            self._write('"""')
            return

        doc = doc.doc.replace('\n', ' ')
        textwidth = 80 - len(self._cur_indent)
        self._write('"""')
        for i in textwrap.wrap(doc, textwidth):
            self._write(i)
        self._write('"""')


    def _on_param_begin(self, p):
        self._cur_param_name = str(p.name)
        self._cur_fields = []
        self._cur_enum_names = []
        self._write('class %s(_ParamDefBase):', p.name, indent=1)
        self._write_doc(p.name)
        self._write('TAG = %d', p.tag)

    def _on_param_end(self, p):
        # gen slots and packer
        self._write('__slots__ = [%s]', ', '.join(
            map('"{.name}"'.format, self._cur_fields)))
        struct_fmt = ''.join(i.fmt for i in self._cur_fields)
        if not struct_fmt:
            struct_fmt = 'x'
        else:
            # add padding at end
            max_t = max(struct_fmt, key=struct.calcsize)
            struct_fmt += '0{}'.format(max_t)
        self._write('_packer = struct.Struct("%s")', struct_fmt)

        # gen __init__ signature
        self._write('def __init__(%s):',
                    ', '.join(['self'] +
                              list('{}={}'.format(i.name, i.default)
                                   for i in self._cur_fields)),
                    indent=1)
        # gen __init__ doc
        self._write('"""')
        for i in self._cur_fields:
            self._write(':type {}: :class:`.{}`'.format(i.name, i.type))
            if i.doc:
                self._write(':param {}: {}'.format(i.name, i.doc))
        self._write('"""')

        # gen cvt in __init__
        for i in self._cur_fields:
            self._write('self.%s = %s', i.name, i.cvt)

        self._unindent()
        self._unindent()
        self._write('')

    def _on_member_enum(self, e):
        qualname = '{}.{}'.format(self._cur_param_name, e.name)

        self._write('class %s(_EnumBase):', e.name, indent=1)
        self._write_doc(e.name)

        for idx, emem in enumerate(e.members):
            self._write('%s = "%s"', emem, emem)
            self._write_doc(emem)
            self._enum_member2num.append('id({}.{}):{}'.format(
                qualname, emem, idx))

        for emem, emem_alis in e.member_alias:
            self._write('%s = %s', emem_alis, emem)

        self._unindent()
        self._write('')

        self._cur_fields.append(self.FieldDef(
            name=e.name_field,
            cvt='{}.convert({})'.format(qualname, e.name_field),
            fmt='I',
            default="'{}'".format(e.members[e.default]),
            type=qualname,
            doc=None))

    def _on_member_enum_alias(self, e):
        self._write('%s = %s.%s', e.name, e.src_class, e.src_name)
        s = e.src_enum
        qualname = '{}.{}'.format(e.src_class, e.src_name)
        self._cur_fields.append(self.FieldDef(
            name=e.name_field,
            cvt='{}.convert({})'.format(qualname, e.name_field),
            fmt='I',
            default="'{}'".format(s.members[e.get_default()]),
            type=qualname,
            doc=None))

    def _get_py_default(self, cppdefault):
        if not isinstance(cppdefault, str):
            return cppdefault

        d = cppdefault
        if d.endswith('f'): # 1.f
            return d[:-1]
        if d.endswith('ull'):
            return d[:-3]
        if d == 'false':
            return 'False'
        if d == 'true':
            return 'True'
        if d.startswith('DTypeEnum::'):
            return '"{}"'.format(d.split(':')[2].lower())
        return d

    def _on_member_field(self, f):
        d = self._get_py_default(f.default)

        self._cur_fields.append(self.FieldDef(
            name=f.name,
            cvt='{}({})'.format(f.dtype.pycvt, f.name),
            fmt=f.dtype.pyfmt,
            default=d,
            type=f.dtype.pycvt,
            doc=f.name.doc
        ))

    def _on_const_field(self, f):
        d = self._get_py_default(f.default)
        self._write_doc(f.name)
        self._write('%s = %s', f.name, d)



class CPPWriter(IndentWriterBase):
    _param_namespace = 'param'

    _ctor_args = None
    """list of (text in func param, var name); func param name must be var name
    appended by an underscore"""
    _non_static_members = None

    def __call__(self, fout, defs):
        super().__call__(fout)
        self._write('// %s', self._get_header())
        self._write('#pragma once')
        self._write('#include "megdnn/dtype.h"')
        self._write('#include <stdint.h>')
        if self._param_namespace == 'param':
            self._write('#include <string.h>')
        self._write('namespace megdnn {')
        self._write('namespace %s {', self._param_namespace)
        self._process(defs)
        self._write('} // namespace megdnn')
        self._write('} // namespace %s', self._param_namespace)
        self._write('// vim: syntax=cpp.doxygen')

    def _write_doc(self, doc):
        assert isinstance(doc, member_defs.Doc)
        if not doc.doc:
            return

        if doc.no_reformat:
            self._write('/*')
            for i in doc.raw_lines:
                self._write('* ' + i)
            self._write('*/')
            return

        doc = doc.doc.replace('\n', ' ')
        textwidth = 80 - len(self._cur_indent) - 4
        if len(doc) <= textwidth:
            self._write('//! ' + doc)
            return

        self._write('/*!')
        for i in textwrap.wrap(doc, textwidth):
            self._write(' * ' + i)
        self._write(' */')

    def _on_param_begin(self, p):
        self._write_doc(p.name)
        self._write('struct %s {', p.name, indent=1)
        self._write('static MEGDNN_CONSTEXPR uint32_t TAG = %du;', p.tag)
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
                    self._write_doc(i.name)
                    self._write('%s%s %s;', i.dtype.cname_attr, i.dtype.cname, i.name)
                else:
                    assert isinstance(i, (member_defs.Enum, member_defs.EnumAlias))
                    self._write('%s %s;', i.name, i.name_field)
            self._write('}; };')
        if self._ctor_args:
            pdefs, varnames = zip(*self._ctor_args)
            self._write('%s(%s) {', p.name, ', '.join(pdefs), indent=1)
            self._write('memset(this, 0, sizeof(*this));')
            for var in varnames:
                self._write('this->%s = %s_;', var, var)
            self._write('}', indent=-1)
        self._write('};\n', indent=-1)

    def _on_member_enum(self, e):
        self._write_doc(e.name)
        self._write('enum class %s: uint32_t {', e.name, indent=1)
        for idx, i in enumerate(e.members):
            self._write_doc(i)
            v = '{} = {}'.format(i, idx)
            if i is not e.members[-1] or e.member_alias:
                v += ','
            self._write(v)
        for mem, alias in e.member_alias:
            self._write('%s = %s,', alias, mem)
        self._write('};', indent=-1)
        self._non_static_members.append(e)
        self._write('static MEGDNN_CONSTEXPR uint32_t %s_NR_MEMBER = %d;',
                    str(e.name).upper(), len(e.members))
        self._add_ctor_args(e.name,
                            '{}::{}'.format(e.name, e.members[e.default]),
                            e.name_field)

    def _on_member_enum_alias(self, e):
        s = e.src_enum
        self._write('using %s = %s::%s;', e.name, e.src_class, e.src_name)
        self._non_static_members.append(e)
        self._write('static MEGDNN_CONSTEXPR uint32_t %s_NR_MEMBER = %d;',
                    str(e.name).upper(), len(s.members))
        self._add_ctor_args(e.name,
                            '{}::{}'.format(e.name,
                                            s.members[e.get_default()]),
                            e.name_field)

    def _on_member_field(self, f):
        self._non_static_members.append(f)
        self._add_ctor_args(f.dtype.cname, f.default, f.name)

    def _on_const_field(self, f):
        self._write_doc(f.name)
        if 'int' in f.dtype.cname:
            self._write('static constexpr %s%s %s = %s;', f.dtype.cname_attr, f.dtype.cname, f.name, f.default)
        else:
            self._write('static const %s%s %s = %s;', f.dtype.cname_attr, f.dtype.cname, f.name, f.default)



class CPPEnumValueWriter(CPPWriter):
    _param_namespace = 'param_enumv'

    def _on_member_enum(self, e):
        self._write_doc(e.name)
        self._write('struct %s {', e.name, indent=1)
        for idx, val in enumerate(e.members):
            self._write_doc(val)
            self._write('static const uint32_t %s = %d;', val, idx)
        for mem, alias in e.member_alias:
            self._write('static const uint32_t %s = %s;', alias, mem)
        self._write('};', indent=-1)


    def _on_member_enum_alias(self, e):
        s = e.src_enum
        self._write('typedef %s::%s %s;', e.src_class, e.src_name, e.name)

    def _on_member_field(self, f):
        pass

    def _on_const_field(self, f):
        pass


class CPPEnumItemWriter(WriterBase):
    _class_name = None
    _enum_name = None
    _enable = False

    def __init__(self, enum_def):
        self._class_name, self._enum_name = enum_def.split(':')

    def __call__(self, fout, defs):
        super().__call__(fout)
        self._process(defs)

    def _on_param_begin(self, p):
        self._enable = p.name == self._class_name

    def _on_member_enum(self, e):
        if self._enable and e.name == self._enum_name:
            for i in e.members:
                self._fout.write('{}\n'.format(i))

class CPPParamJsonFuncWriter(IndentWriterBase):
    _param_namespace = 'param'
    _param_name = None
    _items = None
    def _write_json_item(self, json_cls, field):
        cls2ctype = {
            'NumberInt': 'int64_t',
            'Number': 'double',
            'Bool': 'bool',
        }
        self._items.append('{"%s", json::%s::make(static_cast<%s>(p.%s))},' % (
            field, json_cls, cls2ctype[json_cls], field))


    def __call__(self, fout, defs):
        super().__call__(fout)
        self._write('// %s', self._get_header())
        self._write('// this file can only be included in '
                    'megbrain/src/plugin/impl/opr_footprint.cpp\n'
                    '// please do not include it directly')
        self._write('#include "megdnn/opr_param_defs.h"')
        self._write('#pragma once')
        self._write('using namespace megdnn;')
        self._write('namespace mgb {')
        self._write('namespace opr {')
        self._write('template<class OprParam>')
        self._write('std::shared_ptr<mgb::json::Value> opr_param_to_json(const OprParam &param);')
        self._process(defs)
        self._write('} // namespace opr')
        self._write('} // namespace mgb')
        self._write('\n// vim: syntax=cpp.doxygen')

    def _on_param_begin(self, p):
        self._write('template<>', indent=0)
        self._write(
            'std::shared_ptr<mgb::json::Value> opr_param_to_json(const param::%s &p) {',
            p.name, indent=1)
        self._param_name = 'param::{}'.format(p.name)
        self._items = []

    def _on_param_end(self, p):
        self._write('return json::Object::make({', indent=1)
        for i in self._items:
            self._write(i, indent=0)
        self._write('});', indent=-1)
        self._write('}', indent=-1)

    def _on_member_enum(self, e):
        self._write('auto %s2str = [](const %s::%s arg) -> std::string {',
                    e.name, self._param_name, e.name, indent=1)
        self._write('switch (arg) {', indent=1)
        enum2str = []
        if isinstance(e, member_defs.EnumAlias):
            members = e.src_enum.members
        else:
            members = e.members
        for idx, i in enumerate(members):
            self._write('case %s::%s::%s: return "%s";',
                        self._param_name, e.name, i, i, indent=0)
        self._write('default: mgb_throw(MegBrainError, "Invalid %s::%s:%%d", static_cast<int>(arg));',
                    self._param_name, e.name, indent=0)
        self._write('}', indent=-1)
        self._write('};', indent=-1)
        self._items.append('{"%s", json::String::make(%s2str(p.%s))},' % (
            e.name_field, e.name, e.name_field))

    def _on_member_enum_alias(self, e):
        self._on_member_enum(e)

    def _on_member_field(self, f):
        self._write_json_item(f.dtype.cppjson, f.name)

    def _on_const_field(self, f):
        pass


def main():
    parser = argparse.ArgumentParser(
        'generate opr param defs from description file')
    parser.add_argument('--enumv', action='store_true',
                        help='generate c++03 compatible code which only '
                        'contains enum values')
    parser.add_argument('-t', '--type', choices=['c++', 'py'], default='c++',
                        help='output type')
    parser.add_argument('--write-enum-items',
                        help='write enum item names to output file; argument '
                        'should be given in the CLASS:ENUM format')
    parser.add_argument('--write-cppjson',
                        help='generate megbrain json serialization implemention'
                        'cpp file')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--imperative', action='store_true',
                        help='generate files for imperatvie ')
    args = parser.parse_args()

    for_imperative = args.imperative

    with open(args.input) as fin:
        inputs = fin.read()
        exec(inputs, {'pdef': ParamDef, 'Doc': member_defs.Doc})
        input_hash = hashlib.sha256()
        input_hash.update(inputs.encode(encoding='UTF-8'))
        input_hash = input_hash.hexdigest()

    if args.type == 'py':
        writer = PyWriter(for_imperative=for_imperative)
    else:
        assert args.type == 'c++'
        if args.enumv:
            writer = CPPEnumValueWriter()
        elif args.write_enum_items:
            writer = CPPEnumItemWriter(args.write_enum_items)
        else:
            writer = CPPWriter()
    with open(args.output, 'w') as fout:
        writer.set_input_hash(input_hash)(fout, ParamDef.all_param_defs)

    if args.write_cppjson:
        writer = CPPParamJsonFuncWriter()
        with open(args.write_cppjson, 'w') as fout:
            writer.set_input_hash(input_hash)(fout, ParamDef.all_param_defs)

if __name__ == '__main__':
    main()
