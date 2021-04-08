/**
 * \file dnn/include/megdnn/oprs/base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include <type_traits>
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"

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

/**
 * \brief separate algo by datatype for Matmul and conv
 */
enum class AlgoDataType : uint32_t {
    FLOAT32 = 1 << 0,
    FLOAT16 = 1 << 1,
    QINT8X8X32 = 1 << 2,
    QUINT8X8X32 = 1 << 3,
    INT8X8X16 = 1 << 4,
    INT16X16X32 = 1 << 5,
    INT4X4X16 = 1 << 6,
};

/*!
 * \brief Abstract representation of an algorithm for implementing
 *      the operator
 */
class Algorithm {
public:
    static constexpr uint32_t INVALID_ALGO_TYPE = static_cast<uint32_t>(-1);

    /**
     * \brief the attribe of the algo, such as REPRODUCIBLE, NAIVE
     *
     */
    enum class Attribute : uint32_t {
        /**
         * \brief general algo.
         */
        DEFAULT = 0,

        /**
         * \brief whether the execution result is
         *      reproducible across multiple runs.
         */
        REPRODUCIBLE = 1 << 0,

        /**
         * \brief whether the algo is naive
         * Mark algorithms with simple implementation as NAIVE, so we can filter
         * these algorithms to speed up fastrun.
         * */
        NAIVE = 1 << 1,
    };

    /**
     * \brief Algorithm information, we can get real algo from
     * AlgorithmInfo::Info::Desc
     */
    struct Info {
        struct Desc {
            //! backend of the algo belonging to
            Handle::HandleType handle_type;
            //! indicate the real algo implementation
            uint32_t type = INVALID_ALGO_TYPE;
            //! serialized param of the algo type
            std::string param;
            //! algorithm name
            std::string name;
            bool valid() const { return type != INVALID_ALGO_TYPE; }
            void reset() { type = INVALID_ALGO_TYPE; }

            bool operator==(const Desc& rhs) const {
                return handle_type == rhs.handle_type && type == rhs.type &&
                       param == rhs.param && name == rhs.name;
            }
        } desc;
        Attribute attribute;
        bool valid() const { return desc.valid(); }
        void reset() { desc.reset(); }
        //! desc donate the algo
        bool operator==(const Info& rhs) const { return desc == rhs.desc; }
    };

    virtual ~Algorithm() = default;

    /**
     * \brief get the attribute of the algo
     */
    virtual Attribute attribute() const = 0;

    virtual const char* name() const = 0;
    //! serialized param
    virtual std::string param() const { return {}; }
    virtual uint32_t type() const = 0;

    //! if algo contain all of the attribute in attr
    bool contain_attribute_all(const Attribute& attr) const;

    //! if algo contain any attribute in attr
    bool contain_attribute_any(const Attribute& attr) const;

    void check_attribute(
            const Attribute& positive_attr = Attribute::DEFAULT,
            const Attribute& negative_attr = Attribute::DEFAULT) const;

    static std::string attribute_str(const Attribute& attr);

    Handle::HandleType handle_type() const { return m_handle_type; }

    Info::Desc desc() const { return {handle_type(), type(), param(), name()}; }
    Info info() const {
        return {desc(), attribute()};
    }

    template <typename T>
    static void serialize_write_pod(const T& val, std::string& result) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "type should be trivially copyable");
        static_assert(!std::is_pointer<T>::value,
                      "serialize pointer is unsafe in eager execution mode");
        result.append(reinterpret_cast<const char*>(&val), sizeof(T));
    }

    static void serialize_write_pod(const char* val, std::string& result) {
        result.append(val, strlen(val));
    }

    template <typename T>
    static T deserialize_read_pod(const std::string& data, size_t offset = 0) {
        static_assert(std::is_trivially_copyable<T>::value, "invalid type");
        T ret;
        //! A pointer to an object or incomplete type may be converted to a
        //! pointer to a different object or incomplete type. If the resulting
        //! pointer is not correctly aligned for the pointed-to type, the
        //! behavior is undefined.
        //!
        //! so here we should use memcpy instead of
        //!     *reinterpret_cast<const T*>(&data[offset]);
        memcpy(&ret, data.data() + offset, sizeof(T));
        return ret;
    }

    template <typename T>
    static T deserialize_read_pod(const char* data, size_t offset = 0) {
        static_assert(std::is_trivially_copyable<T>::value, "invalid type");
        T ret;
        //! A pointer to an object or incomplete type may be converted to a
        //! pointer to a different object or incomplete type. If the resulting
        //! pointer is not correctly aligned for the pointed-to type, the
        //! behavior is undefined.
        //!
        //! so here we should use memcpy instead of
        //!     *reinterpret_cast<const T*>(&data[offset]);
        memcpy(&ret, data + offset, sizeof(T));
        return ret;
    }

    enum class OprType : uint32_t {
        MATRIX_MUL_FORWARD,
        BATCHED_MATRIX_MUL_FORWARD,
        CONVOLUTION_FORWARD,
        CONVOLUTION_BACKWARD_DATA,
        CONVOLUTION_BACKWARD_FILTER,
        CONVOLUTION3D_FORWARD,
        CONVOLUTION3D_BACKWARD_DATA,
        CONVOLUTION3D_BACKWARD_FILTER,
        LOCAL_SHARE_FORWARD,
        LOCAL_SHARE_BACKWARD_DATA,
        LOCAL_SHARE_BACKWARD_FILTER,
        DEFORMABLE_CONV_FORWARD,
        DEFORMABLE_CONV_BACKWARD_DATA,
        DEFORMABLE_CONV_BACKWARD_FILTER,
        CONVBIAS_FORWARD,
        BATCH_CONV_FORWARD,
    };

    struct SearchItem {
        OprType opr_type;
        //! serialized param
        std::string param;
        TensorLayoutArray layouts;
    };

    /**
     * \brief get subopr list of the algo
     *
     * \param layouts origin layouts of the parent opr
     * \param opr parent opr
     */
    virtual std::vector<SearchItem> get_subopr_list(const TensorLayoutArray&,
                                                    const OperatorBase*) const {
        return {};
    }

protected:
    Handle::HandleType m_handle_type = Handle::HandleType::NAIVE;
};

MEGDNN_DEF_ENUM_CLASS_BIT_OPR(Algorithm::Attribute)

//! policy for executing the operator
struct ExecutionPolicy {
    //! INVALID_ALGO_TYPE algo_type means using heuristic
    Algorithm::Info::Desc algo;
    std::vector<ExecutionPolicy> sub_policy;
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
    using AlgorithmInfo = detail::Algorithm::Info;
    using AlgorithmDesc = detail::Algorithm::Info::Desc;
    using Algorithm = detail::Algorithm;

    /*!
     * \brief get a string representation for current algorithm set;
     *
     * get_all_algorithms() may return different algorithms only if
     * algorithm set name differs. This is used for checking cache
     * validity.
     */
    virtual const char* get_algorithm_set_name() const = 0;

    ExecutionPolicy& execution_policy() { return m_execution_policy; }

    const ExecutionPolicy& execution_policy() const {
        return m_execution_policy;
    }

    virtual Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) = 0;

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
    using AlgorithmInfo = detail::Algorithm::Info;
    using AlgoAttribute = detail::Algorithm::Attribute;

    //! get all possible algorithm decriptions for the specified layouts
    std::vector<AlgorithmInfo> get_all_algorithms_info(const TensorLayout& p0,
                                                       const TensorLayout& p1,
                                                       const TensorLayout& p2) {
        std::vector<AlgorithmInfo> ret;
        for (auto&& algo : get_all_algorithms(p0, p1, p2)) {
            ret.emplace_back(algo->info());
        }
        return ret;
    }

    /**
     * \brief Returns the best algorithm information which indicate the
     * algorithm by heuristic.
     *
     * The selected algorithm should not use workspace more than
     * \p workspace_limit_in_bytes.
     */
    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2,
            size_t workspace_limit_in_bytes =
                    std::numeric_limits<size_t>::max(),
            const AlgoAttribute& positive_attr = AlgoAttribute::DEFAULT,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
        return get_algorithm_heuristic(p0, p1, p2, workspace_limit_in_bytes,
                                       positive_attr, negative_attr)
                ->info();
    }

protected:
    ~MultiAlgoOpr() = default;

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
            const AlgoAttribute& positive_attr = AlgoAttribute::DEFAULT,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) = 0;
};

//! specializae for nargs == 4
template <class Opr>
class MultiAlgoOpr<Opr, 4> : public MultiAlgoOpr<Opr, -1> {
public:
    using Algorithm = detail::Algorithm;
    using AlgorithmInfo = detail::Algorithm::Info;
    using AlgoAttribute = detail::Algorithm::Attribute;

    //! get all possible algorithm decriptions for the specified layouts
    std::vector<AlgorithmInfo> get_all_algorithms_info(const TensorLayout& p0,
                                                       const TensorLayout& p1,
                                                       const TensorLayout& p2,
                                                       const TensorLayout& p3) {
        std::vector<AlgorithmInfo> ret;
        for (auto&& algo : get_all_algorithms(p0, p1, p2, p3)) {
            ret.emplace_back(algo->info());
        }
        return ret;
    }

    /**
     * \brief Returns the best algorithm information which indicate the
     * algorithm by heuristic.
     *
     * The selected algorithm should not use workspace more than
     * \p workspace_limit_in_bytes.
     */
    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            size_t workspace_limit_in_bytes =
                    std::numeric_limits<size_t>::max(),
            const AlgoAttribute& positive_attr = AlgoAttribute::DEFAULT,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
        return get_algorithm_heuristic(p0, p1, p2, p3, workspace_limit_in_bytes,
                                       positive_attr, negative_attr)
                ->info();
    }

protected:
    ~MultiAlgoOpr() = default;

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
            const AlgoAttribute& positive_attr = AlgoAttribute::DEFAULT,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) = 0;
};

//! specializae for nargs == 5
template <class Opr>
class MultiAlgoOpr<Opr, 5> : public MultiAlgoOpr<Opr, -1> {
public:
    using Algorithm = detail::Algorithm;
    using AlgorithmInfo = detail::Algorithm::Info;
    using AlgoAttribute = detail::Algorithm::Attribute;

    //! get all possible algorithm decriptions for the specified layouts
    std::vector<AlgorithmInfo> get_all_algorithms_info(const TensorLayout& p0,
                                                       const TensorLayout& p1,
                                                       const TensorLayout& p2,
                                                       const TensorLayout& p3,
                                                       const TensorLayout& p4) {
        std::vector<AlgorithmInfo> ret;
        for (auto&& algo : get_all_algorithms(p0, p1, p2, p3, p4)) {
            ret.emplace_back(algo->info());
        }
        return ret;
    }

    /**
     * \brief Returns the best algorithm information which indicate the
     * algorithm by heuristic.
     *
     * The selected algorithm should not use workspace more than
     * \p workspace_limit_in_bytes.
     */
    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            const TensorLayout& p4,
            size_t workspace_limit_in_bytes =
                    std::numeric_limits<size_t>::max(),
            const AlgoAttribute& positive_attr = AlgoAttribute::DEFAULT,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
        return get_algorithm_heuristic(p0, p1, p2, p3, p4,
                                       workspace_limit_in_bytes, positive_attr,
                                       negative_attr)
                ->info();
    }

protected:
    ~MultiAlgoOpr() = default;

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
            const AlgoAttribute& positive_attr = AlgoAttribute::DEFAULT,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) = 0;
};

//! specializae for nargs == 8
template <class Opr>
class MultiAlgoOpr<Opr, 8> : public MultiAlgoOpr<Opr, -1> {
public:
    using Algorithm = detail::Algorithm;
    using AlgorithmInfo = detail::Algorithm::Info;
    using AlgoAttribute = detail::Algorithm::Attribute;

    //! get all possible algorithm decriptions for the specified layouts
    std::vector<AlgorithmInfo> get_all_algorithms_info(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            const TensorLayout& p4, const TensorLayout& p5,
            const TensorLayout& p6, const TensorLayout& p7) {
        std::vector<AlgorithmInfo> ret;
        for (auto&& algo : get_all_algorithms(p0, p1, p2, p3, p4, p5, p6, p7)) {
            ret.emplace_back(algo->info());
        }
        return ret;
    }

    /**
     * \brief Returns the best algorithm information which indicate the
     * algorithm by heuristic.
     *
     * The selected algorithm should not use workspace more than
     */
    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& p0, const TensorLayout& p1,
            const TensorLayout& p2, const TensorLayout& p3,
            const TensorLayout& p4, const TensorLayout& p5,
            const TensorLayout& p6, const TensorLayout& p7,
            size_t workspace_limit_in_bytes =
                    std::numeric_limits<size_t>::max(),
            const AlgoAttribute& positive_attr = AlgoAttribute::DEFAULT,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
        return get_algorithm_heuristic(p0, p1, p2, p3, p4, p5, p6, p7,
                                       workspace_limit_in_bytes, positive_attr,
                                       negative_attr)
                ->info();
    }

protected:
    ~MultiAlgoOpr() = default;

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
            const AlgoAttribute& positive_attr = AlgoAttribute::DEFAULT,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) = 0;
};
}  // namespace detail

using Algorithm = detail::Algorithm;
using AlgoAttribute = Algorithm::Attribute;
using ExecutionPolicy = detail::ExecutionPolicy;
}  // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
