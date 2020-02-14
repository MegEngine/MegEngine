/**
 * \file python_module/src/cpp/megbrain_config.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#ifndef SWIG

#pragma once

#include "megbrain_build_config.h"
#include "./megbrain_wrap.h"
#include <Python.h>
using mgb::cg::SymbolVar;
#endif

//! wrap by a class so swig can put the functions in a namespace
class _config {
    public:
        static bool set_comp_graph_option(
                CompGraph &cg, const std::string &name, int val_int);

        static bool comp_graph_is_eager(CompGraph &cg);

        static void add_extra_vardep(
                const SymbolVar &var, const SymbolVar &dep);

        static void begin_set_opr_priority(
                CompGraph &cg, int priority);
        static void end_set_opr_priority(CompGraph &cg);

        static void begin_set_exc_opr_tracker(
                CompGraph &cg, PyObject *tracker);
        static void end_set_exc_opr_tracker(CompGraph &cg);

        //! return (opr_msg, fwd tracker, grad tracker) or None
        static PyObject* get_opr_tracker(CompGraph &cg, size_t var_id);

        static void set_opr_sublinear_memory_endpoint(const SymbolVar &var);

        static void set_fork_cuda_warning_flag(int flag);

        static bool is_cuda_ctx_set();

        //! get cuda gencode strings for local devices
        static std::string get_cuda_gencode();

        //! get cuda lib paths.
        static std::vector<std::string> get_cuda_lib_path();

        //! get cuda include paths.
        static std::vector<std::string> get_cuda_include_path();

        //! get cuda version
        static int get_cuda_version();

        static bool is_compiled_with_cuda();

        static void load_opr_library(
                const char* self_path, const char* lib_path);

        static std::vector<std::pair<uint64_t, std::string>>
            dump_registered_oprs();

#if MGB_ENABLE_OPR_MM
        static int create_mm_server(const std::string& server_addr, int port);

        static void group_barrier(const std::string& server_addr,
                int port, uint32_t size, uint32_t rank);
#endif
};

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
