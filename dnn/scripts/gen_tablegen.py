#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import textwrap
import os
import hashlib
import struct
import io

from gen_param_defs import member_defs, ParamDef, IndentWriterBase


class ConverterWriter(IndentWriterBase):
    _skip_current_param = False
    _last_param = None
    _current_tparams = None
    _packed = None
    _const = None

    def __call__(self, fout, defs):
        super().__call__(fout)
        self._write("// %s", self._get_header())
        self._write("#ifndef MGB_PARAM")
        self._write("#define MGB_PARAM")
        self._process(defs)
        self._write("#endif // MGB_PARAM")

    def _ctype2attr(self, ctype, value):
        if ctype == 'uint32_t':
            return 'MgbUI32Attr', value
        if ctype == 'uint64_t':
            return 'MgbUI64Attr', value
        if ctype == 'int32_t':
            return 'MgbI32Attr', value
        if ctype == 'float':
            return 'MgbF32Attr', value
        if ctype == 'double':
            return 'MgbF64Attr', value
        if ctype == 'bool':
            return 'MgbBoolAttr', value
        if ctype == 'DTypeEnum':
            self._packed = False
            return 'MgbDTypeAttr', 'megdnn::DType::from_enum(megdnn::{})'.format(value)
        raise RuntimeError("unknown ctype")

    def _on_param_begin(self, p):
        self._last_param = p
        if p.is_legacy:
            self._skip_current_param = True
            return
        self._packed = True
        self._current_tparams = []
        self._const = set()

    def _on_param_end(self, p):
        if self._skip_current_param:
            self._skip_current_param = False
            return
        if self._packed:
            self._write("class {0}ParamBase<string accessor> : MgbPackedParamBase<\"{0}\", accessor> {{".format(p.name), indent=1)
        else:
            self._write("def {0}Param: MgbParamBase<\"{0}\"> {{".format(p.name), indent=1)
        self._write("let fields = (ins", indent=1)
        self._write(",\n{}".format(self._cur_indent).join(self._current_tparams))
        self._write(");", indent=-1)
        self._write("}\n", indent=-1)
        if self._packed:
            self._write("def {0}Param : {0}ParamBase<\"param\">;\n".format(p.name))
        self._current_tparams = None
        self._packed = None
        self._const = None

    def _wrapped_with_default_value(self, attr, default):
        return 'MgbDefaultValuedAttr<{}, \"{}\">'.format(attr, default)

    def _on_member_enum(self, e):
        p = self._last_param

        # Note: always generate llvm Record def for enum attribute even it was not
        # directly used by any operator, or other enum couldn't alias to this enum
        td_class = "{}{}".format(p.name, e.name)
        fullname = "::megdnn::param::{}".format(p.name)
        enum_def = "MgbEnumAttr<\"{}\", \"{}\", [".format(fullname, e.name)
        def format(v):
            return '\"{}\"'.format(str(v))
        enum_def += ','.join(format(i) for i in e.members)
        enum_def += "]>"
        self._write("def {} : {};".format(td_class, enum_def))

        if self._skip_current_param:
            return

        # wrapped with default value
        default_val = "static_cast<{}::{}>({})".format(fullname, e.name, e.default)
        wrapped = self._wrapped_with_default_value(td_class, default_val)

        self._current_tparams.append("{}:${}".format(wrapped, e.name_field))

    def _on_member_enum_alias(self, e):
        p = self._last_param
        if self._skip_current_param:
            return

        # write enum attr def
        td_class = "{}{}".format(p.name, e.name)
        fullname = "::megdnn::param::{}".format(p.name)
        base_td_class = "{}{}".format(e.src_class, e.src_name)
        enum_def = "MgbEnumAliasAttr<\"{}\", \"{}\", {}>".format(fullname, e.name, base_td_class)
        self._write("def {} : {};".format(td_class, enum_def))

        # wrapped with default value
        default_val = "static_cast<{}::{}>({})".format(fullname, e.name, e.get_default())
        wrapped = self._wrapped_with_default_value(td_class, default_val)

        self._current_tparams.append("{}:${}".format(wrapped, e.name_field))


    def _on_member_field(self, f):
        if self._skip_current_param:
            return
        attr, value = self._ctype2attr(f.dtype.cname, str(f.default))
        if str(value) in self._const:
            value = '::megdnn::param::{}::{}'.format(self._last_param.name, value)
        wrapped = self._wrapped_with_default_value(attr, value)
        self._current_tparams.append("{}:${}".format(wrapped, f.name))

    def _on_const_field(self, f):
        self._const.add(str(f.name))

def main():
    parser = argparse.ArgumentParser('generate op param tablegen file')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    with open(args.input) as fin:
        inputs = fin.read()
        exec(inputs, {'pdef': ParamDef, 'Doc': member_defs.Doc})
        input_hash = hashlib.sha256()
        input_hash.update(inputs.encode(encoding='UTF-8'))
        input_hash = input_hash.hexdigest()

    writer = ConverterWriter()
    with open(args.output, 'w') as fout:
        writer.set_input_hash(input_hash)(fout, ParamDef.all_param_defs)

if __name__ == "__main__":
    main()
