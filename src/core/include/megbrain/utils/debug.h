/**
 * \file src/core/include/megbrain/utils/debug.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/exception.h"
#include "megbrain/tensor.h"

#include <string>
#include <vector>

namespace mgb {
namespace debug {
#if MGB_ENABLE_DEBUG_UTIL

    class ForkAfterCudaError final: public SystemError {
        public:
            using SystemError::SystemError;

            //! function to throw this exception; could be overwritten
            static void(*throw_)();
    };

    struct BacktraceResult {
        std::vector<std::pair<const char*, size_t>> stack;

        /*!
         * \brief format and write to dst
         */
        void fmt_to_str(std::string &dst);
    };

    /*!
     * \brief get (file name, address) pairs for backtracing
     * \param nr_exclude number of frames to be excluded
     */
     BacktraceResult backtrace(int nr_exclude = 1);

    /*!
     * \brief set cuda fork warning flag
     * \param flag
     *      0: disable warning
     *      1: log warning message
     *      2: throw ForkAfterCudaError() exception
     */
    void set_fork_cuda_warning_flag(int flag);

    /*!
     * \brief supress fork warning in this scope
     *
     * A warning would be printed when calling fork() after CUDA context has
     * been initialized. Include this class in the scope to supress the warning.
     */
    class ScopedForkWarningSupress {
        static std::atomic_size_t sm_depth;

    public:
        ScopedForkWarningSupress() { ++sm_depth; }
        ~ScopedForkWarningSupress() { --sm_depth; }

        static bool supress() { return sm_depth.load() != 0; }
    };

#endif  // MGB_ENABLE_DEBUG_UTIL

    /*!
     * \brief dump tensor dtype, value and name to a single binary
     *
     * The binary can be parsed by `megbrain.plugin.load_tensor_binary` python
     * function
     */
    std::string dump_tensor(const HostTensorND& value, const std::string& name);

    static inline std::string dump_tensor(const DeviceTensorND& value,
                                          const std::string& name) {
        return dump_tensor(HostTensorND{}.copy_from(value).sync(), name);
    }

    //! write the value of a string to file
    void write_to_file(const char* filename, const std::string& content,
                       const char* mode = "wb");

    /*!
     * \brief check whether absolute/relative error for each element is not
     *      greater than \p maxerr
     * \return None if tensors are considered equal; or a human-readable
     *      message indicating their difference
     */
    Maybe<std::string> compare_tensor_value(
            const HostTensorND &expect, const char *expect_expr,
            const HostTensorND &get, const char *get_expr,
            float maxerr);

} // namespace debug
} // namespace mgb


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
