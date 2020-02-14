/**
 * \file python_module/src/cpp/plugin.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief helpers for debugging
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */


#ifndef SWIG

#pragma once

#include "./megbrain_wrap.h"

#include "megbrain/plugin/profiler.h"
#include "megbrain/plugin/infkern_finder.h"
#include "megbrain/plugin/num_range_checker.h"
#include "megbrain/plugin/opr_io_dump.h"

#endif // SWIG

#include <Python.h>

class _CompGraphProfilerImpl {
#ifndef SWIG
    std::shared_ptr<mgb::ComputingGraph> m_comp_graph;
    mgb::GraphProfiler m_profiler;
#endif

    public:
        _CompGraphProfilerImpl(CompGraph &cg):
            m_comp_graph{cg.get().shared_from_this()},
            m_profiler{m_comp_graph.get()}
        {
        }

        std::string _get_result() {
            auto json = m_profiler.to_json_full(
                    m_comp_graph->current_comp_seq());
            return json->to_string();
        }
};

class _NumRangeCheckerImpl {
#ifndef SWIG
    std::shared_ptr<mgb::ComputingGraph> m_comp_graph;
    mgb::NumRangeChecker m_checker;
#endif

    public:
        _NumRangeCheckerImpl(CompGraph &cg, float range):
            m_comp_graph{cg.get().shared_from_this()},
            m_checker{m_comp_graph.get(), range}
        {
        }
};

class _TextOprIODumpImpl {
#ifndef SWIG
    std::shared_ptr<mgb::ComputingGraph> m_comp_graph;
    mgb::TextOprIODump m_dump;
#endif

    public:
        _TextOprIODumpImpl(CompGraph &cg, const char *fpath):
            m_comp_graph{cg.get().shared_from_this()},
            m_dump{m_comp_graph.get(), fpath}
        {
        }

        void _print_addr(bool flag) {
            m_dump.print_addr(flag);
        }

        void _max_size(size_t size) {
            m_dump.max_size(size);
        }
};

class _BinaryOprIODumpImpl {
#ifndef SWIG
    std::shared_ptr<mgb::ComputingGraph> m_comp_graph;
    mgb::BinaryOprIODump m_dump;
#endif

    public:
        _BinaryOprIODumpImpl(CompGraph &cg, const char *fpath):
            m_comp_graph{cg.get().shared_from_this()},
            m_dump{m_comp_graph.get(), fpath}
        {
        }
};

class _InfkernFinderImpl {
#ifndef SWIG
    static size_t sm_id;
    const size_t m_id;
    std::shared_ptr<mgb::ComputingGraph> m_comp_graph;
    mgb::InfkernFinder m_finder;
    mgb::InfkernFinder::InputValueRecord::FullRecord m_inp_val;
#endif

    public:
        _InfkernFinderImpl(CompGraph &cg, bool record_input_value);

        size_t _write_to_file(const char *fpath);

        size_t _get_input_values_prepare(size_t opr_id);
        const char* _get_input_values_var_name(size_t idx);
        size_t _get_input_values_var_idx(size_t idx);
        size_t _get_input_values_run_id(size_t idx);
        CompGraphCallbackValueProxy  _get_input_values_val(size_t idx);

        std::string __repr__();

};

class _FastSignal {
#ifndef SWIG
    class Impl;
    static Impl sm_impl;

    static void signal_hander(int signum);
#endif
    public:
        static void register_handler(int signum, PyObject *func);
        static void shutdown();
};

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

