/**
 * \file src/jit/include/megbrain/jit/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/metahelper.h"
#include "megbrain_build_config.h"

#if MGB_JIT

#include <string>
#include <utility>
#include <vector>

namespace mgb {
namespace jit {
namespace str_util {

using StrReplaceMap = std::vector<std::pair<std::string, std::string>>;

/*!
 * replace all non-overlapping occurrences of the given (from,to) pairs in-place
 * in text, where each (from,to) replacement pair is processed in the order it
 * is given.
 */
void replace_all_pairs_inplace(std::string& text, const StrReplaceMap& replace);

//! append new (k, v) pairs to a replace map
static inline void append_replace_map(
        StrReplaceMap& map,
        std::initializer_list<std::pair<std::string, std::string>> newitems) {
    for (auto&& i : newitems) {
        map.push_back(std::move(i));
    }
}

template <typename T>
inline std::string str(T i) {
    return std::to_string(i);
}

}  // namespace str_util

/*!
 * \brief OS-agnostic abstraction to manage executable files
 *
 * Note that the folder to store object and shared library files are managed
 * globally, so the \p name arguments in all APIs should only contain the file
 * name without directory.
 *
 * All the methods should use exception for error reporting. The return values
 * are always non-null.
 *
 * The intermediate files would be removed unless MGB_JIT_KEEP_INTERM is set
 */
class ExecutableHelper : public NonCopyableObj {
protected:
    ~ExecutableHelper() = default;

public:
    //! load shared library, like dlopen()
    virtual void* load_lib(const std::string& name) = 0;

    //! resolve a function in a library, like dlsym()
    virtual void* resolve_func(void* handle, const std::string& func_name) = 0;

    //! unload a library, like dlclose()
    virtual void unload_lib(void* handle) = 0;

    /*!
     * \brief compile C++ source code to object file
     *
     * Note the output file name would be modified to include hash of the
     * source, so multiple version can co-exist. The output file would be kept
     * regardless of MGB_JIT_KEEP_INTERM setting.
     *
     * The name `secondary` originates from .SECONDARY target of GNU make.
     *
     * \param out_name output filename template; it should not include the .cpp
     *      suffix
     *
     * \return object file name (without dir path)
     */
    virtual std::string compile_cpp_source_secondary(const char* source,
                                                     const char* out_name) = 0;

    //! link object files to shared library
    virtual void link(const SmallVector<std::string>& inp_names,
                      const std::string& out_name) = 0;

    //! remove a file in the working dir
    virtual void remove(const std::string& name) = 0;

    //! get real path of a file in the working dir
    virtual std::string realpath(const std::string& name) = 0;

    //! remove file if MGB_JIT_KEEP_INTERM is not set
    void remove_interm(const std::string& name) {
        if (!keep_interm()) {
            remove(name);
        }
    }

    //! link to library and load
    void* link_and_load(const SmallVector<std::string>& inp_names,
                        const std::string& out_name) {
        link(inp_names, out_name);
        return load_lib(out_name);
    }

    //! resolve function and write to target pointer
    template <typename T>
    void resolve_func(T& dst, void* handle, const std::string& func_name) {
        dst = reinterpret_cast<T>(resolve_func(handle, func_name));
    }

    //! write content to file
    void write_file(const std::string& name, const std::string& data);

    //! whether MGB_JIT_KEEP_INTERM is set
    static bool keep_interm();

    //! get the singleton instance
    static ExecutableHelper& get();
};

//! get name for next kernel to be compiled; guaranteed to be globally unique
//! in this process
std::string next_kernel_name();

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
