/**
 * \file src/gopt/impl/layout_transform_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./utils.h"
#include "megbrain/gopt/global_layout_transform.h"

using namespace mgb;
using namespace gopt;

/* ================= LayoutTransformContext ==================*/
LayoutTransformContext& LayoutTransformContext::add_opr_config(
        Typeinfo* opr, OprFormat opr_format) {
    auto& dispatchers = m_opr_configs[opr];
    dispatchers[opr_format] =
            OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                    opr, opr_format);
    return *this;
}

LayoutTransformContext& LayoutTransformContext::add_opr_config(
        Typeinfo* opr, SmallVector<OprFormat> opr_formats) {
    auto& dispatchers = m_opr_configs[opr];
    for (auto opr_fmt : opr_formats) {
        dispatchers[opr_fmt] =
                OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                        opr, opr_fmt);
    }
    return *this;
}

// vim: syntax=cpp.doxygen
