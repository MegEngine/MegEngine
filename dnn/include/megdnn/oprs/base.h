/**
 * \file dnn/include/megdnn/oprs/base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/basic_types.h"

#include "megdnn/internal/visibility_prologue.h"
namespace megdnn {

class Handle;

/**
 * \brief base class for all operators
 *
 * This is an helper class. Users should not use OperatorBase directly.
 * Operators should be created by handle->create_opr<>().
 *
 * Each operator must provides the following constexpr values:
 *
 *  * NR_INPUTS: number of input vars
 *  * NR_OUTPUTS: number of output vars
 *  * OPERATOR_TYPE: operator type as an enum
 *
 * If the operator has dynamic inputs or in_out param, the corresponding
 * NR_INPUTS is -1.
 *
 * For an operator whose NR_INPUTS >= 0 and NR_OUTPUTS >= 0, the operator must
 * also provide following methods:
 *
 *  * void exec(_megdnn_in inputs..., _megdnn_tensor_out outputs...,
 *              _megdnn_workspace workspace)
 *  * void deduce_layout(const TensorLayout& inputs...,
 *                       TensorLayout& outputs...)
 *  * size_t get_workspace_in_bytes(const TensorLayout &inputs...,
 *                                  const TensorLayout &outputs)
 */
class OperatorBase {
public:
    explicit OperatorBase(Handle* handle) : m_handle(handle) {}
    virtual ~OperatorBase();

    //! get the handle from which this operator is created
    Handle* handle() const { return m_handle; }

    //! whether this opr guarantees that its exec() is thread-safe
    virtual bool is_thread_safe() const { return false; }

    /*!
     * \brief set the tracker to be used with MegcoreAsyncErrorInfo
     *
     * Most operators do not have async errors so this function has a
     * default empty implementation.
     */
    virtual void set_error_tracker(void*) {}

private:
    Handle* m_handle;
};

namespace detail {
/**
 * \brief AlgoSelectionStrategy is the advance information for selecting
 * algo
 */
enum class AlgoSelectionStrategy {
    HEURISTIC = 0,  //!< heristic to select the algos
    FAST_RUN = 1,
    FULL_RUN = 2,
};

/*!
 * \brief Abstract representation of an algorithm for implementing
 *      the operator
 *
 * All pointers to Algorithm should be allocated globally and usable
 * across multiple megdnn handles, and they should not be freed by
 * the caller.
 */
class Algorithm {
public:
    /**
     * \brief whether the execution result is
     *      reproducible across multiple runs.
     */
    virtual bool is_reproducible() const = 0;
    virtual const char* name() const = 0;

    //! a pointer to represent class type
    virtual void* type() const { return nullptr; }

protected:
    ~Algorithm() = default;
};

/*!
 * \brief define Algorithm and ExecutionPolicy for oprs that have
 *      multiple impl algos
 *
 * \tparam Opr the operator class
 * \tparam nargs number of arguments
 */
template <class Opr, int nargs>
class MultiAlgoOpr;

//! base def
template <class Opr>
class MultiAlgoOpr<Opr, -1> {
public:
    using Algorithm = detail::Algorithm;
    /*!
     * \brief get a string representation for current algorithm set;
     *
     * get_all_algorithms() may return different algorithms only if
     * algorithm set name differs. This is used for checking cache
     * validity.
     */
    virtual const char* get_algorithm_set_name() const = 0;

    //! policy for executing the operator
    struct ExecutionPolicy {
        //! nullptr means using heuristic
        Algorithm* algorithm = nullptr;
    };

    ExecutionPolicy& execution_policy() { return m_execution_policy; }

    const ExecutionPolicy& execution_policy() const {
        return m_execution_policy;
    }

protected:
    ~MultiAlgoOpr() = default;

private:
    ExecutionPolicy m_execution_policy;
};

//! specialize for nargs == 3
template <class Opr>
class MultiAlgoOpr<Opr, 3> : public MultiAlgoOpr<Opr, -1> {
public:
    using Algorithm = detail::Algorithm;

    //! get all possible algorithms for the specified layouts
    virtual std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2) = 0;

    /**
     * \brief Returns the best algorithm by heuristic.
     *
     * The selected algorithm should not use workspace more than
     * \p workspace_limit_in_bytes.
     */
    virtual Algorithm* get_algorithm_heuristic(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2,
            size_t workspace_limit_in_bytes =
                    std::numeric_limits<size_t>::max(),
            bool reproducible = false) = 0;

protected:
    ~MultiAlgoOpr() = default;
};

//! specializae for nargs == 4
template <class Opr>
class MultiAlgoOpr<Opr, 4> : public MultiAlgoOpr<Opr, -1> {
public:
    using Algorithm = detail::Algorithm;

    //! get all possible algorithms for the specified layouts
    virtual std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3) = 0;

    /**
     * \brief Returns the best algorithm by heuristic.
     *
     * The selected algorithm should not use workspace more than
     * \p workspace_limit_in_bytes.
     */
    virtual Algorithm* get_algorithm_heuristic(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            size_t workspace_limit_in_bytes =
                    std::numeric_limits<size_t>::max(),
            bool reproducible = false) = 0;

protected:
    ~MultiAlgoOpr() = default;
};

//! specializae for nargs == 5
template <class Opr>
class MultiAlgoOpr<Opr, 5> : public MultiAlgoOpr<Opr, -1> {
public:
    using Algorithm = detail::Algorithm;

    //! get all possible algorithms for the specified layouts
    virtual std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            const TensorLayout& p4) = 0;

    /**
     * \brief Returns the best algorithm by heuristic.
     *
     * The selected algorithm should not use workspace more than
     * \p workspace_limit_in_bytes.
     */
    virtual Algorithm* get_algorithm_heuristic(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            const TensorLayout& p4,
            size_t workspace_limit_in_bytes =
                    std::numeric_limits<size_t>::max(),
            bool reproducible = false) = 0;

protected:
    ~MultiAlgoOpr() = default;
};

//! specializae for nargs == 8
template <class Opr>
class MultiAlgoOpr<Opr, 8> : public MultiAlgoOpr<Opr, -1> {
public:
    using Algorithm = detail::Algorithm;

    //! get all possible algorithms for the specified layouts
    virtual std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            const TensorLayout& p4, const TensorLayout& p5,
            const TensorLayout& p6, const TensorLayout& p7) = 0;

    /**
     * \brief Returns the best algorithm by heuristic.
     *
     * The selected algorithm should not use workspace more than
     * \p workspace_limit_in_bytes.
     */
    virtual Algorithm* get_algorithm_heuristic(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            const TensorLayout& p4, const TensorLayout& p5,
            const TensorLayout& p6, const TensorLayout& p7,
            size_t workspace_limit_in_bytes =
                    std::numeric_limits<size_t>::max(),
            bool reproducible = false) = 0;

protected:
    ~MultiAlgoOpr() = default;
};
}  // namespace detail
}  // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
