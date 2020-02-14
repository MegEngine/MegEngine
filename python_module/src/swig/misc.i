/*
 * $File: misc.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */


%{
#include "megbrain/utils/persistent_cache.h"
#include "megbrain/serialization/helper.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/plugin/opr_footprint.h"
using _PyStackExtracter = PyStackExtracter;
using _PersistentCache = mgb::PersistentCache;
using _PersistentCacheBlob = _PersistentCache::Blob;
using _MaybePersistentCacheBlob = mgb::Maybe<_PersistentCacheBlob>;
using _OptimizeForInferenceOptions = mgb::gopt::OptimizeForInferenceOptions;
%}

%feature("director") _PyStackExtracter;
class _PyStackExtracter {
    public:
        virtual ~_PyStackExtracter() = default;
        virtual std::string extract() = 0;
        static void reg(_PyStackExtracter *p);
};

// from Blob to python bytes
%typemap(in) const _PersistentCacheBlob& {
    mgb_assert(PyBytes_Check($input));
    $1->ptr = PyBytes_AsString($input);
    $1->size = PyBytes_Size($input);
}
%typemap(directorin) const _PersistentCacheBlob& {
    $input = PyBytes_FromStringAndSize(
        static_cast<const char*>($1.ptr), $1.size);
}
%typemap(directorout) _MaybePersistentCacheBlob {
    mgb_assert($1->ob_refcnt >= 2, "persistent cache result refcnt too small");
    if ($1 == Py_None) {
        $result = mgb::None;
    } else {
        mgb_assert(PyBytes_Check($input));
        _PersistentCacheBlob blob;
        blob.ptr = PyBytes_AsString($1);
        blob.size = PyBytes_Size($1);
        $result = blob;
    }
}

%feature("director") _PersistentCache;
class _PersistentCache {
    public:
        virtual ~_PersistentCache() = default;

        virtual void put(const std::string &category,
                const _PersistentCacheBlob &key,
                const _PersistentCacheBlob &value) = 0;

        virtual _MaybePersistentCacheBlob get(
                const std::string &category,
                const _PersistentCacheBlob &key) = 0;

        %extend {
            static void reg(_PersistentCache *p) {
                _PersistentCache::set_impl({p, [](_PersistentCache*){}});
            }
        }
};

struct _OptimizeForInferenceOptions {
#define SET(n) void enable_##n()
    SET(f16_io_f32_comp);
    SET(f16_io_comp);
    SET(fuse_conv_bias_nonlinearity);
    SET(use_nhwcd4);
    SET(use_tensor_core);
    SET(fuse_conv_bias_with_z);
    SET(use_nchw88);
#undef SET
};

%inline {
    static SymbolVarArray _optimize_for_inference(
            const SymbolVarArray& dest_vars,
            const _OptimizeForInferenceOptions& opt) {
        return mgb::gopt::optimize_for_inference(dest_vars, opt);
    }

    // defined in function_replace.cpp
    void _register_logger(PyObject *logger);
    void _timed_func_set_fork_exec_path(const char *arg0, const char *arg1);
    void _timed_func_exec_cb(const char *user_data);

    // defined in megbrain_wrap.cpp
    void _mgb_global_finalize();
    std::vector<size_t> _get_mgb_version();
    SymbolVarArray _grad(SymbolVar target, SymbolVarArray wrts,
            bool warn_mid_wrt, int use_virtual_grad,
            bool return_zero_for_nodep);
    SymbolVar _inter_graph_trans_var(
            CompGraph &dest_graph, SymbolVar src);
    SymbolVar _get_graph_optimizer_replaced_var(SymbolVar src);
    void _add_update_fastpath(SharedND& dest, SharedND& delta,
            float alpha, float beta, float bias);
    void _add_update_fastpath(SharedND& dest,
            CompGraphCallbackValueProxy& delta,
            float alpha, float beta, float bias);

    static SymbolVar _current_grad_target(CompGraph &graph) {
        return mgb::cg::current_grad_target(graph.get());
    }

    uint32_t _get_dtype_num(PyObject *dtype) {
        return static_cast<uint32_t>(npy::dtype_np2mgb(dtype).enumv());
    }

    PyObject* _get_serialized_dtype(PyObject *dtype) {
        std::string sdtype;
        auto write = [&sdtype](const void* data, size_t size) {
            auto pos = sdtype.size();
            sdtype.resize(pos + size);
            memcpy(&sdtype[pos], data, size);
        };
        mgb::serialization::serialize_dtype(npy::dtype_np2mgb(dtype), write);
        return PyBytes_FromStringAndSize(sdtype.data(), sdtype.size());
    }

    size_t max_size_t() {
        return std::numeric_limits<size_t>::max();
    }

    std::string _get_opr_fp_graph_exec(
        CompGraph& cg, const SymbolVarArray& outputs) {
        auto json = mgb::OprFootprint::get_opr_fp_graph_exec(cg.get(), outputs);
        return json->to_string();
    }
}

// vim: ft=swig
