/**
 * \file src/core/impl/graph/event.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph/event.h"
#include "./cg_impl.h"

using namespace mgb::cg::event;

MGB_TYPEINFO_OBJ_IMPL(OprInserted);
MGB_TYPEINFO_OBJ_IMPL(OprExecStart);
MGB_TYPEINFO_OBJ_IMPL(AfterWait);
MGB_TYPEINFO_OBJ_IMPL(OprExecKernelStart);
MGB_TYPEINFO_OBJ_IMPL(OprExecKernelEnd);
MGB_TYPEINFO_OBJ_IMPL(OprExecFinished);
MGB_TYPEINFO_OBJ_IMPL(BeforeKernel);
MGB_TYPEINFO_OBJ_IMPL(AfterKernel);
MGB_TYPEINFO_OBJ_IMPL(StaticMemAlloc);
MGB_TYPEINFO_OBJ_IMPL(CompSeqOrderDetermined);
MGB_TYPEINFO_OBJ_IMPL(CompSeqExecBeforeStart);
MGB_TYPEINFO_OBJ_IMPL(CompSeqExecFinished);
MGB_TYPEINFO_OBJ_IMPL(CompSeqExecError);
MGB_TYPEINFO_OBJ_IMPL(SubgraphAssociated);
#if MGB_ENABLE_VAR_DEV_MEM_DEFRAGMENTER
MGB_TYPEINFO_OBJ_IMPL(BeforeMemDefrag);
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

