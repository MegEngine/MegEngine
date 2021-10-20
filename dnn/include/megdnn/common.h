/**
 * \file dnn/include/megdnn/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"

#if MGB_ENABLE_GETENV
#define MGB_GETENV ::std::getenv
#else
#define MGB_GETENV(_name) static_cast<char*>(nullptr)
#endif

#ifdef WIN32
#define unsetenv(_name)                _putenv_s(_name, "");
#define setenv(name, value, overwrite) _putenv_s(name, value)
#endif

namespace megdnn {

/*!
 * \brief whether there is an algorithm from algo_pack() that is available for
 * current size
 */
template <class Opr, typename... Args>
bool has_available_algo(Opr* opr, Args&&... args) {
    const typename Opr::AlgoBase::SizeArgs size_args(opr, std::forward<Args>(args)...);
    for (auto i : Opr::algo_pack().all_algos) {
        if (i->is_available(size_args)) {
            return true;
        }
    }
    return false;
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
