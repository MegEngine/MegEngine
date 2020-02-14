/**
 * \file python_module/src/cpp/megbrain_serialize.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */


#ifndef SWIG

#pragma once

#include "megbrain/serialization/serializer.h"
#include "./megbrain_wrap.h"
using mgb::cg::SymbolVar;
using mgb::cg::SymbolVarArray;
#endif

#ifdef SWIG
%feature("autodoc",
"An object that is passed to the callback in \
:func:`.serialize_comp_graph_to_file`.") TensorValueDumperContext;
%feature("autodoc",
"An object that is passed to the callback in \
:func:`.load_comp_graph_from_file`.") TensorValueLoaderContext;
%feature("director") _TensorValueDumperCallback;
%feature("director") _TensorValueLoaderCallback;
%feature("director") _CompNodeMapperCallback;

%template(_VectorPairStringSymbolVar) std::vector<std::pair<std::string, SymbolVar>>;
%typemap(directorout) PyObjRefKeeper {
    Py_XINCREF($input);
    $result = PyObjRefKeeper($input);
}
#endif
class TensorValueDumperContext {
#ifndef SWIG
    mgb::serialization::OutputFile &m_fout;
    const mgb::cg::OperatorNodeBase &m_opr;
    const mgb::HostTensorND &m_value;
#endif

    public:
        TensorValueDumperContext() = delete;
        TensorValueDumperContext(const TensorValueDumperContext&) = delete;
        TensorValueDumperContext& operator = (
                const TensorValueDumperContext&) = delete;

#ifndef SWIG
        TensorValueDumperContext(
                mgb::serialization::OutputFile &fout,
                const mgb::cg::OperatorNodeBase &opr,
                const mgb::HostTensorND &value):
            m_fout{fout}, m_opr{opr}, m_value{value}
        {
        }
#endif
        ~TensorValueDumperContext() noexcept;

        const char* _name() const {
            return m_opr.cname();
        }

        const char* _type() const {
            return m_opr.dyn_typeinfo()->name;
        }

        PyObject* _value();

        void _write(PyObject *bytes);

        void _write_default() {
            mgb::serialization::GraphDumpConfig::default_tensor_value_dumper(
                    m_fout, m_opr, m_value);
        }

#ifdef SWIG
%include "./megbrain_serialize_TensorValueDumperContext.py"
#endif

};

class TensorValueLoaderContext {
#ifndef SWIG
    const mgb::TensorLayout &m_layout;
    mgb::serialization::InputFile &m_fin;
#endif

    public:
        TensorValueLoaderContext() = delete;
        TensorValueLoaderContext(const TensorValueLoaderContext&) = delete;
        TensorValueLoaderContext& operator=(const TensorValueLoaderContext&) =
                delete;

#ifndef SWIG
        TensorValueLoaderContext(const mgb::TensorLayout &layout,
                                 mgb::serialization::InputFile &fin)
                : m_layout(layout), m_fin(fin) {}
#endif
        ~TensorValueLoaderContext() noexcept;

        std::vector<size_t> _get_shape() const;
        PyObject* _get_dtype() const;

        // Returns bytes
        PyObject* _read(size_t n);

#ifdef SWIG
%include "./megbrain_serialize_TensorValueLoaderContext.py"
#endif
};

class _TensorValueDumperCallback {
    public:
        virtual ~_TensorValueDumperCallback() = default;
        virtual void call(TensorValueDumperContext &ctx) = 0;
};

class _TensorValueLoaderCallback {
    public:
        virtual ~_TensorValueLoaderCallback() = default;
        virtual PyObjRefKeeper call(TensorValueLoaderContext &ctx) = 0;
};

class _CompNodeMapperCallback {
    public:
        virtual ~_CompNodeMapperCallback() = default;
        virtual std::string call(const std::string &desc) = 0;
};

#ifdef SWIG
%include "megbrain/serialization/dump_format.h"
#else
#include "megbrain/serialization/dump_format.h"
#endif

void _serialize_comp_graph_to_file(
        const char *fpath, bool append,
        mgb::serialization::GraphDumpFormat format,
        const SymbolVarArray &output_vars,
        int keep_var_name, bool keep_param_name, bool keep_opr_priority,
        _TensorValueDumperCallback *tensor_value_dumper,
        std::vector<size_t> &stat,
        std::vector<std::string> &inputs,
        std::vector<std::string> &outputs,
        std::vector<std::string> &params);

std::string _get_info_for_strip(const SymbolVarArray &dest_vars);

CompGraph _load_comp_graph_from_file(
        const char *fpath, _CompNodeMapperCallback *cn_mapper,
        _TensorValueLoaderCallback *tensor_value_loader,
        /* Outputs */
        std::vector<std::pair<std::string, SymbolVar>> &output_var_map,
        SymbolVarArray &output_var_list);

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
