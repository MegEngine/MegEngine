/**
 * \file python_module/src/cpp/opr_helper.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief helper for wrapping special oprs
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "./megbrain_wrap.h"

#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#if MGB_ENABLE_OPR_MM
#include "megbrain/opr/collective_comm.h"
#endif

using AxisIndexer = mgb::opr::indexing::AxisIndexer;

/*!
 * \brief wrapping callbacks used for opr::Split::Options::make_callback
 */
class _SplitPartCallback {
    bool m_cb_created = false;

    public:
        virtual ~_SplitPartCallback() = default;
        virtual std::vector<size_t> call(size_t tot_size) = 0;

        using callback_t = mgb::opr::Split::Options::callback_t;
        callback_t make_callback();
};

class _SetGradCallback {
    bool m_cb_created = false;

    public:
        virtual ~_SetGradCallback() = default;
        virtual mgb::SymbolVar call(CompGraph &graph) = 0;
        virtual bool empty() = 0;

        using callback_t = mgb::opr::SetGrad::GradGetter;
        callback_t make_callback();
};

/*!
 * \brief wrapping callbacks used for subclasses of opr::RemoteIOBase
 */
class _TimeoutCallback {
    bool m_cb_created = false;

    public:
        virtual ~_TimeoutCallback() = default;
        /*!
         * \brief Will be overrided by swig generated code, calls into Python.
         */
        virtual bool call() = 0;

        using callback_t = mgb::thin_function<bool()>;
        callback_t make_callback();
};

#if MGB_ENABLE_OPR_MM
mgb::opr::CollectiveComm::Param load_collective_comm_params(
        PyObject* params, mgb::ComputingGraph* graph);
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
