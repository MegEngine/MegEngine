/**
 * \file src/core/include/megbrain/exception.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain_build_config.h"

#include <memory>
#include <stdexcept>
#include <string>

#if MGB_ENABLE_EXCEPTION
#define MGB_IF_EXCEPTION(x...) x
#else
#define MGB_IF_EXCEPTION(x...)
#endif

#if (defined(__GNUC__) && !defined(__ANDROID__) && !defined(ANDROID) && \
     !defined(__APPLE__))
#include <cxxabi.h>  // for abi::__forced_unwind
#define __MGB_HANDLE_FORCED_UNWIND MGB_CATCH(abi::__forced_unwind&, { throw; })
#else
#define __MGB_HANDLE_FORCED_UNWIND
#endif

/*!
 * \brief catch all exceptions and store in an exception_ptr; usually used in
 *      worker threads
 *
 * This macro should be inserted after a try block
 * \param _scope_msg const char* type text to describe where this exception is
 *      caught
 */
#define MGB_CATCH_ALL_EXCEPTION(_scope_msg, _ptr)                       \
    MGB_CATCH(std::exception& _exc, {                                   \
        mgb_log_error("caught exception in %s; what(): %s", _scope_msg, \
                      _exc.what());                                     \
        _ptr = std::current_exception();                                \
    })                                                                  \
    __MGB_HANDLE_FORCED_UNWIND                                          \
    MGB_CATCH(..., {                                                    \
        mgb_log_error("caught unknown exception in %s", _scope_msg);    \
        _ptr = std::current_exception();                                \
    })                                                                  \
    do {                                                                \
    } while (0)

/*!
 * \brief catch all exceptions in a class destructor and log error and abort
 *
 * \param _scope_msg const char* type text to describe where this exception is
 *      caught
 */
#define MGB_HANDLE_EXCEPTION_DTOR(_scope_msg)                                 \
    MGB_CATCH(std::exception& _exc, {                                         \
        mgb_log_error("abort due to exception in %s; what(): %s", _scope_msg, \
                      _exc.what());                                           \
        abort();                                                              \
    })                                                                        \
    MGB_CATCH(..., {                                                          \
        mgb_log_error("abort due to unknown exception in %s", _scope_msg);    \
    })                                                                        \
    do {                                                                      \
    } while (0)

namespace mgb {

//! the most general MegBrain exception type; also base class for all megbrain
//! exceptions
class MegBrainError : public std::exception {
protected:
    std::string m_msg;

public:
    /*!
     * \brief base class for extra information to be associated with an
     *      exception
     */
    class ExtraInfo {
    public:
        virtual ~ExtraInfo() = default;
    };

    MegBrainError(const std::string& msg) : m_msg(msg) { init(); }

    const char* what() const noexcept override { return m_msg.c_str(); }

    /*!
     * \brief get associated extra info, or nullptr
     */
    const ExtraInfo* extra_info() const { return m_extra_info.get(); }

    /*!
     * \brief set extra info
     */
    template <typename T>
    MegBrainError& extra_info(T&& ptr) {
        m_extra_info = ptr;
        return *this;
    }

    ~MegBrainError() noexcept = default;

private:
    std::shared_ptr<ExtraInfo> m_extra_info;
    void init();
};

//! base class for system error: error caused by uncontrollable environment
class SystemError : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};

/*!
 * \brief exception to be thrown if failing to allocate memory
 */
class MemAllocError : public SystemError {
public:
    using SystemError::SystemError;
};

class CudaError final : public SystemError {
public:
    /*!
     * \brief get extra info for current cuda status, to be appended in
     *      error message
     */
    static std::string get_cuda_extra_info();
    CudaError(const std::string& msg);
};

class AtlasError final: public SystemError {
public:
    AtlasError(const std::string& msg);
};


class ROCmError final : public SystemError {
public:
    /*!
     * \brief get extra info for current rocm status, to be appended in
     *      error message
     */
    static std::string get_rocm_extra_info();

    ROCmError(const std::string& msg);
};

class CnrtError final : public SystemError {
public:
    /*!
     * \brief get extra info for current cnrt status, to be appended in
     * error message
     */
    static std::string get_cnrt_extra_info();

    CnrtError(const std::string& msg);
};

class CndevError final : public SystemError {
public:
    CndevError(const std::string& msg);
};

class CnmlError final : public SystemError {
public:
    CnmlError(const std::string& msg);
};

class AssertionError final : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};

//! datatype conversion error
class ConversionError final : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};

class TensorCopyOverlapError final : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};

class TensorReshapeError final : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};

class SerializationError final : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};

class MegDNNError final : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};

//! megbrain internal error; should be treated as a bug
class InternalError final : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};

class TimeoutError final : public MegBrainError {
public:
    using MegBrainError::MegBrainError;
};


}  // namespace mgb

namespace mgb {

bool has_uncaught_exception();

}  // namespace mgb


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
