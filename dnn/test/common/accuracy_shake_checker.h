/**
 * \file dnn/test/common/accuracy_shake_checker.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <vector>
#include "megdnn/oprs.h"
#include "src/common/conv_bias.h"
#include "src/common/utils.h"
#include "test/common/checker.h"
#include "test/common/index.h"

namespace megdnn {
namespace test {

namespace {

template <class Opr>
struct BatchTrait {
    //! index of batch in tensor, 3 for CHWN4 e.g.
    static size_t index_of_batch(const typename Opr::Param&) { return 0; }

    //! indices contain batch in inputs and outputs, src(0) dst(2) for conv e.g.
    static std::vector<size_t> indices_contain_batch;

    static std::vector<size_t> indices_contain_batch_broadcast;
};

template <class Opr>
std::vector<size_t> BatchTrait<Opr>::indices_contain_batch = {};
template <class Opr>
std::vector<size_t> BatchTrait<Opr>::indices_contain_batch_broadcast  = {};

#define DEFAULT_INDEX_OF_BATCH(opr) \
    static size_t index_of_batch(const opr::Param&) { return 0; }

#define CONV_INDEX_OF_BATCH(opr)                        \
    static size_t index_of_batch(const opr::Param& p) { \
        if (p.format == opr::Param::Format::CHWN4) {    \
            return 3;                                   \
        }                                               \
        return 0;                                       \
    }

#define OPR_WITHOUT_INPUT_BROADCAST(INDEX_OF_BATCH, opr, idxs, idxs_brdcst) \
    template <>                                                             \
    struct BatchTrait<opr> {                                                \
        INDEX_OF_BATCH(opr)                                                 \
        static std::vector<size_t> indices_contain_batch;                   \
        static std::vector<size_t> indices_contain_batch_broadcast;         \
    };                                                                      \
    std::vector<size_t> BatchTrait<opr>::indices_contain_batch = idxs;      \
    std::vector<size_t> BatchTrait<opr>::indices_contain_batch_broadcast =  \
            idxs_brdcst;

OPR_WITHOUT_INPUT_BROADCAST(DEFAULT_INDEX_OF_BATCH,
                            megdnn::Convolution3DForward,
                            (std::initializer_list<size_t>{0, 2}), {})
OPR_WITHOUT_INPUT_BROADCAST(DEFAULT_INDEX_OF_BATCH,
                            megdnn::Convolution3DBackwardData,
                            (std::initializer_list<size_t>{1, 2}), {})
OPR_WITHOUT_INPUT_BROADCAST(DEFAULT_INDEX_OF_BATCH,
                            megdnn::Convolution3DBackwardFilter,
                            (std::initializer_list<size_t>{0, 1}), {})
OPR_WITHOUT_INPUT_BROADCAST(DEFAULT_INDEX_OF_BATCH, megdnn::BatchedMatrixMul,
                            (std::initializer_list<size_t>{0, 1, 2}), {})

OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH, megdnn::ConvolutionForward,
                            (std::initializer_list<size_t>{0, 2}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH,
                            megdnn::ConvolutionBackwardData,
                            (std::initializer_list<size_t>{1, 2}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH,
                            megdnn::ConvolutionBackwardFilter,
                            (std::initializer_list<size_t>{0, 1}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH, megdnn::LocalShareForward,
                            (std::initializer_list<size_t>{0, 2}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH, megdnn::LocalShareBackwardData,
                            (std::initializer_list<size_t>{1, 2}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH,
                            megdnn::LocalShareBackwardFilter,
                            (std::initializer_list<size_t>{0, 1}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH, megdnn::DeformableConvForward,
                            (std::initializer_list<size_t>{0, 2, 3, 4}), {})
OPR_WITHOUT_INPUT_BROADCAST(
        CONV_INDEX_OF_BATCH, megdnn::DeformableConvBackwardData,
        (std::initializer_list<size_t>{0, 2, 3, 4, 5, 6, 7}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH,
                            megdnn::DeformableConvBackwardFilter,
                            (std::initializer_list<size_t>{0, 1, 2, 3}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH, megdnn::BatchConvBiasForward,
                            (std::initializer_list<size_t>{0, 1, 2, 3, 4}), {})
OPR_WITHOUT_INPUT_BROADCAST(CONV_INDEX_OF_BATCH, megdnn::ConvBiasForward,
                            (std::initializer_list<size_t>{0, 3, 4}), {2})
#undef OPR_WITHOUT_INPUT_BROADCAST
#undef DEFAULT_INDEX_OF_BATCH
#undef CONV_INDEX_OF_BATCH

template <class Opr>
struct LayoutsModifier {
    static void on(TensorLayoutArray& layouts, const typename Opr::Param& p,
                   size_t new_batch_size) {
        size_t batch_index = BatchTrait<Opr>::index_of_batch(p);
        for (size_t index : BatchTrait<Opr>::indices_contain_batch) {
            layouts.at(index)[batch_index] = new_batch_size;
        }

        for (size_t index : BatchTrait<Opr>::indices_contain_batch_broadcast) {
            if (!check_bias_share_in_channel(layouts.at(index), p.format)) {
                layouts.at(index)[batch_index] = new_batch_size;
            }
        }
    }
};

#define OPR_NO_BIAS(opr)                                                      \
    template <>                                                               \
    struct LayoutsModifier<opr> {                                             \
        static void on(TensorLayoutArray& layouts,                            \
                       const typename opr::Param& p, size_t new_batch_size) { \
            size_t batch_index = BatchTrait<opr>::index_of_batch(p);          \
            for (size_t index : BatchTrait<opr>::indices_contain_batch) {     \
                layouts.at(index)[batch_index] = new_batch_size;              \
            }                                                                 \
        }                                                                     \
    };

OPR_NO_BIAS(megdnn::Convolution3D)
OPR_NO_BIAS(megdnn::BatchedMatrixMul)
#undef OPR_NO_BIAS

template <>
struct LayoutsModifier<megdnn::MatrixMul> {
public:
    static void on(TensorLayoutArray& layouts,
                   const megdnn::MatrixMul::Param& p,
                   size_t new_batch_size) {
        assert(!p.transposeA && !p.transposeB);
        MEGDNN_MARK_USED_VAR(p);
        layouts.at(0)[0] = new_batch_size;
        layouts.at(2)[0] = new_batch_size;
    }
};

template <class Opr, typename OprAlgoProxy = OprAlgoProxy<Opr>>
class AlgoGenerator {
public:
    AlgoGenerator(ExecutionPolicyAlgoName name)
            : m_policy_name{name} {}

    std::vector<Algorithm::Info::Desc> operator()(
            Opr* opr, const CheckerHelper::TensorValueArray& arr) {
        TensorLayoutArray layouts;
        for (auto&& val : arr) {
            layouts.push_back(val.layout);
        }
        std::vector<Algorithm::Info::Desc> ret;
        megdnn_assert(layouts.size() == OprTrait<Opr>::arity);
        auto vec = AlgoProxy<Opr, OprTrait<Opr>::arity>::get_all_algorithms_info(
                     opr, layouts);
        for (auto algo_info : vec) {
            if (!(algo_info.attribute &
                  AlgoAttribute::ACCURACY_DEPEND_ON_BATCH) &&
                (algo_info.attribute & AlgoAttribute::REPRODUCIBLE) &&
                std::regex_match(
                        algo_info.desc.name,
                        std::regex("(.*)(" + m_policy_name.name + ")(.*)"))) {
                ret.push_back(algo_info.desc);
            } else {
                continue;
            }
        }
        return ret;
    }

private:
    ExecutionPolicyAlgoName m_policy_name;
};

}  // namespace

::testing::AssertionResult __assert_tensor_binary_eq(
        const char* expr0, const char* expr1, const char* expr2,
        const TensorND& v0, const TensorND& v1,
        const Algorithm::Info::Desc& algo);

template <typename Opr, typename Proxy = OprProxy<Opr>>
class AccuracyShakeChecker : public CheckerHelper {
public:
    static constexpr int arity_in = OprArityTrait<Opr>::arity_in;
    using Param = typename Opr::Param;
    using BeforeExecCallback = std::function<std::vector<Algorithm::Info::Desc>(
            Opr*, const TensorValueArray&)>;
    AccuracyShakeChecker(Handle* handle, bool check_dispatch = false)
            : CheckerHelper(handle, check_dispatch),
              m_before_exec_callback{AlgoGenerator<Opr>("")},
              m_param(Param()) {}

    TensorLayoutArray make_layouts(const TensorShapeArray& shapes) {
        TensorLayoutArray layouts(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            DType dt = (m_dtype.find(i) != m_dtype.end() ? m_dtype[i]
                                                         : dtype::Float32());
            TensorFormat fmt =
                    (m_fmt.find(i) != m_fmt.end() ? m_fmt[i] : TensorFormat{});
            layouts[i] = TensorLayout(shapes[i], dt, fmt);
        }
        return layouts;
    }

    /*!
     * \brief execute opr on current param/dtype/rng config
     * \param shapes input/output shapes, which would be passed as
     *      arguments to Opr::deduce_layout
     *
     * Checker would construct TensorLayout vectors from shapes and dtypes,
     * and call exec(TensorLayoutArray &).
     */
    AccuracyShakeChecker& exec(const TensorShapeArray& shapes) {
        exec(make_layouts(shapes));
        return *this;
    }

    void exec(TensorLayoutArray layouts);

    AccuracyShakeChecker& set_param(Param p) {
        m_param = p;
        opr()->param() = p;
        return *this;
    }
    AccuracyShakeChecker& set_dtype(size_t idx, DType dtype) {
        m_dtype[idx] = dtype;
        return *this;
    }
    AccuracyShakeChecker& set_rng(size_t idx, RNG* rng) {
        m_rng[idx] = rng;
        return *this;
    }

    //! set a callback to be invoked before executing the operator
    AccuracyShakeChecker& set_before_exec_callback(
            const BeforeExecCallback& cb) {
        m_before_exec_callback = cb;
        return *this;
    }

    AccuracyShakeChecker& reset_before_exec_callback() {
        m_before_exec_callback = nullptr;
        return *this;
    }

    //! get the opr impl so setting other than param() can be modified
    Opr* opr() {
        if (!m_opr_cur) {
            m_opr_cur = m_handle_cur->create_operator<Opr>();
        }
        return m_opr_cur.get();
    }

private:
    BeforeExecCallback m_before_exec_callback;
    Param m_param;
    Proxy m_proxy;
    std::unique_ptr<Opr> m_opr_cur;
    std::shared_ptr<TensorValueArray> m_tensors_cur_host,
            m_tensors_single_batch_host;

    void init_host_values();

    void check_tensors_ignore_batch(
            const TensorValueArray& tensors_single_batch,
            const TensorValueArray& tensors, const Algorithm::Info::Desc& desc);
};

template <typename Opr, typename Proxy>
void AccuracyShakeChecker<Opr, Proxy>::exec(TensorLayoutArray layouts) {
    auto opr_cur = this->opr();
    opr_cur->param() = m_param;

    m_proxy.deduce_layout(opr_cur, layouts);

    TensorLayoutArray layouts_single_batch = layouts;
    for (size_t i=0; i<layouts_single_batch.size(); ++i) {
        ASSERT_TRUE(layouts[i].is_physical_contiguous())
                << "layouts should be physical contiguous "
                << layouts[i].to_string();
    }

    ASSERT_TRUE(0 == BatchTrait<Opr>::index_of_batch(opr_cur->param()))
                << "index of batch should be 0 ";

    LayoutsModifier<Opr>::on(layouts_single_batch, opr_cur->param(), 1);

    // allocate input
    auto tensors_single_batch_storage =
            alloc_tensors(m_handle_cur, layouts_single_batch, 0);
    m_tensors_single_batch_host =
            alloc_tensors(m_handle_naive.get(), layouts_single_batch, 0);
    auto tensors_cur_storage = alloc_tensors(m_handle_cur, layouts, 0);
    m_tensors_cur_host =
            alloc_tensors(m_handle_naive.get(), layouts, 0);
    auto &&tensors_single_batch = *tensors_single_batch_storage;
    auto &&tensors_single_batch_host = *m_tensors_single_batch_host;
    auto &&tensors_cur = *tensors_cur_storage;
    auto &&tensors_cur_host = *m_tensors_cur_host;

    // allocate output
    auto tensors_single_batch_storage_out =
            alloc_tensors(m_handle_naive.get(), layouts_single_batch, 0);
    auto tensors_cur_storage_out =
            alloc_tensors(m_handle_naive.get(), layouts, 0);
    auto &&tensors_single_batch_out = *tensors_single_batch_storage_out;
    auto &&tensors_cur_out = *tensors_cur_storage_out;

    init_host_values();

    copy_tensors_to_device(tensors_cur, tensors_cur_host);
    copy_tensors_to_device(tensors_single_batch, tensors_single_batch_host);

    std::vector<Algorithm::Info::Desc> algo_desc;
    if (m_before_exec_callback) {
        algo_desc = m_before_exec_callback(opr_cur, tensors_cur);
    } else {
        algo_desc.push_back({});
    }
    for (size_t i = 0; i < algo_desc.size(); ++i) {
        opr_cur->execution_policy().algo = algo_desc[i];
        m_proxy.exec(opr_cur, tensors_cur);
        m_proxy.exec(opr_cur, tensors_single_batch);

        copy_tensors_from_device(tensors_cur_out, tensors_cur);
        copy_tensors_from_device(tensors_single_batch_out,
                                 tensors_single_batch);

        check_tensors_ignore_batch(tensors_single_batch_out, tensors_cur_out,
                                   algo_desc[i]);
    }
}

template <typename Opr, typename Proxy>
void AccuracyShakeChecker<Opr, Proxy>::init_host_values() {
    size_t index_of_batch = 0;
    auto &&tensors_single_batch = *m_tensors_single_batch_host;
    auto &&tensors_cur = *m_tensors_cur_host;
    for (size_t i = 0; i < arity_in; ++i) {
        auto &&tensor_single_batch = tensors_single_batch[i];
        auto &&tensor_cur = tensors_cur[i];
        auto rng = m_rng[i];
        if (!rng)
            rng = m_default_rng.get();
        rng->gen(tensor_single_batch);

        dt_byte* raw_storage_cur = static_cast<dt_byte*>(tensor_cur.raw_ptr) +
                                   tensor_cur.layout.span().low_byte;
        dt_byte* raw_storage_single_batch =
                static_cast<dt_byte*>(tensor_single_batch.raw_ptr) +
                tensor_single_batch.layout.span().low_byte;
        const size_t step = tensor_single_batch.layout.span().dist_byte();
        if (tensor_cur.layout.eq_shape(tensor_single_batch.layout)) {
            memcpy(raw_storage_cur, raw_storage_single_batch, step);
        } else {
            ASSERT_TRUE(1 == tensor_single_batch.layout[index_of_batch])
                << "bad batch size "
                << tensor_single_batch.layout[index_of_batch];
            for (size_t b=0; b<tensor_cur.layout[index_of_batch]; ++b) {
                memcpy(raw_storage_cur, raw_storage_single_batch, step);
                raw_storage_cur += step;
            }
        }
    }
}

template <typename Opr, typename Proxy>
void AccuracyShakeChecker<Opr, Proxy>::check_tensors_ignore_batch(
        const TensorValueArray& tensors_single_batch,
        const TensorValueArray& tensors, const Algorithm::Info::Desc& algo) {
    for (size_t i = 0; i < tensors_single_batch.size(); ++i) {
        if (tensors_single_batch[i].layout.ndim == 0 ||
            tensors_single_batch[i].layout.eq_shape(tensors[i].layout))
            continue;
        ASSERT_PRED_FORMAT3(::megdnn::test::__assert_tensor_binary_eq,
                            tensors_single_batch[i], tensors[i], algo);
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
