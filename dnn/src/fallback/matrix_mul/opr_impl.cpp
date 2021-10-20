/**
 * \file dnn/src/fallback/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/matrix_mul/opr_impl.h"
#include <unordered_map>

#include "megdnn/oprs/base.h"
#include "src/common/algo_chooser.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/algos.h"
#include "src/fallback/matrix_mul/gemm_impl.h"
#include "src/fallback/matrix_mul/generic_strategy.h"
#include "src/naive/handle.h"
#include "src/naive/matrix_mul/opr_impl.h"

#if MEGDNN_X86
#include "src/x86/matrix_mul/opr_impl.h"
#elif MEGDNN_AARCH64
#include "src/aarch64/matrix_mul/opr_impl.h"
#elif MEGDNN_ARMV7
#include "src/armv7/matrix_mul/opr_impl.h"
#endif

using namespace megdnn;
using namespace fallback;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoF32K8x12x1 f32_k8x12x1;
    AlgoGemv gemv;
    AlgoNaive naive;
    SmallVector<AlgoBase*> m_all_algos;
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack() {
        m_all_algos.emplace_back(&gemv);
        m_all_algos.emplace_back(&f32_k8x12x1);
        m_all_algos.emplace_back(&naive);
        for (auto&& algo : m_all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }

    const SmallVector<AlgoBase*>& all_algos() const { return m_all_algos; }
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

const MatrixMulImpl::AlgoPack& MatrixMulImpl::algo_pack() {
    static AlgoPack algo_pack;
    return algo_pack;
}

SmallVector<MatrixMulImpl::AlgoBase*> MatrixMulImpl::get_all_packed_algo() {
    return algo_pack().all_algos();
}

SmallVector<MatrixMulImpl::AlgoBase*> MatrixMulImpl::select_algo_type(
        AlgoTypePack index) {
    megdnn_assert(
            nr_type_contain(index.data_type),
            "Matmul algo selection only support one type");
    SmallVector<MatrixMulImpl::AlgoBase*> algos;
    for (auto&& algo : get_all_packed_algo()) {
        auto algo_desc = algo->matmul_description();
        if (contain_data_type(algo_desc.algo_type.data_type, index.data_type) &&
            algo_desc.algo_type.format == index.format) {
            algos.push_back(algo);
        }
    }
    return algos;
}

std::vector<MatrixMul::Algorithm*> MatrixMulImpl::get_all_algorithms(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    std::vector<Algorithm*> gemm_algos, gemv_algos;
    auto kern_size_param = make_kern_size_param(A, B, C);
    for (auto&& algo : get_all_packed_algo()) {
        if (algo->usable(kern_size_param)) {
            if (algo->algoset() == AlgoBase::AlgoSet::ALGO_TYPE_GEMV) {
                // simple gemv
                gemv_algos.push_back(algo);
            } else {
                gemm_algos.push_back(algo);
            }
        }
    }
    gemv_algos.insert(gemv_algos.end(), gemm_algos.begin(), gemm_algos.end());
    return gemv_algos;
}

std::vector<MatrixMul::Algorithm*> MatrixMulImpl::get_all_algorithms_safe(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    auto gemv_algos_safe = get_all_algorithms(A, B, C);
    megdnn_assert(!gemv_algos_safe.empty(), "no usable MatrixMul fwd algorithm");
    return gemv_algos_safe;
}

MatrixMulImpl::Algorithm* MatrixMulImpl::get_algorithm_from_desc(
        const AlgorithmDesc& desc) {
    if (!desc.valid()) {
        return nullptr;
    } else {
        switch (desc.handle_type) {
            case Handle::HandleType::FALLBACK: {
                const auto& map = algo_pack().all_algos_map();
                megdnn_assert(map.find(desc) != map.end());
                return map.at(desc);
            };

#if MEGDNN_X86
            case Handle::HandleType::X86:
                return x86::MatrixMulImpl::get_algo_from_desc(desc);
#elif MEGDNN_AARCH64 || MEGDNN_ARMV7
            case Handle::HandleType::ARM_COMMON:
                return arm_common::MatrixMulImpl::get_algo_from_desc(desc);
#if MEGDNN_AARCH64
            case Handle::HandleType::AARCH64:
                return aarch64::MatrixMulImpl::get_algo_from_desc(desc);
#else
            case Handle::HandleType::ARMV7:
                return armv7::MatrixMulImpl::get_algo_from_desc(desc);
#endif
#endif
            default:
                megdnn_throw("Unknown handle type");
                return {};
        }
    }
}

MatrixMul::Algorithm* MatrixMulImpl::get_algorithm_heuristic(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    auto kern_size_param = make_kern_size_param(A, B, C);
    if (auto algo = static_cast<AlgoBase*>(
                get_algorithm_from_desc(execution_policy().algo))) {
        megdnn_assert(algo->get_workspace(kern_size_param) < workspace_limit_in_bytes);
        auto cur = megdnn::get_algo_match_attribute<MatrixMulImpl>(
                algo, positive_attr, negative_attr);
        if (cur)
            return cur;
        megdnn_throw(ssprintf(
                "require algorithm without attribute(%s) with "
                "attribute(%s), but given algorithm with "
                "attribute(%s)",
                Algorithm::attribute_str(negative_attr).c_str(),
                Algorithm::attribute_str(positive_attr).c_str(),
                Algorithm::attribute_str(algo->attribute()).c_str()));
    }
    AlgoTypePack algo_type;
    algo_type.data_type = kern_size_param.deduce_algo_data_type();
    algo_type.format = kern_size_param.format;
    auto algos = select_algo_type(algo_type);
    Algorithm* heuristic_algo = nullptr;
    Algorithm* usable_algo = nullptr;
    for (auto&& algo : algos) {
        if (static_cast<AlgoBase*>(algo)->usable(kern_size_param) &&
            static_cast<AlgoBase*>(algo)->get_workspace(kern_size_param) <=
                    workspace_limit_in_bytes) {
            if (static_cast<AlgoBase*>(algo)->preferred_attribute(
                        kern_size_param, positive_attr, negative_attr)) {
                //! use gemv algo if it's prefered
                if (algo->algoset() == AlgoBase::AlgoSet::ALGO_TYPE_GEMV) {
                    return algo;
                } else if (!heuristic_algo) {
                    heuristic_algo = algo;
                }
            } else if (!usable_algo) {
                usable_algo = algo;
            }
        }
    }
    if (!heuristic_algo)
        heuristic_algo = usable_algo;
    megdnn_assert(heuristic_algo, "No usable algorithm found");
    return heuristic_algo;
}

MatrixMulImpl::KernSizeParam MatrixMulImpl::make_kern_size_param(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    KernSizeParam kern_size_param;
    kern_size_param.A_type = A.dtype;
    kern_size_param.B_type = B.dtype;
    kern_size_param.C_type = C.dtype;
    kern_size_param.M = C.shape[0];
    kern_size_param.N = C.shape[1];
    kern_size_param.K = A[1 - param().transposeA];
    kern_size_param.LDA = A.stride[0];
    kern_size_param.LDB = B.stride[0];
    kern_size_param.LDC = C.stride[0];
    kern_size_param.trA = param().transposeA;
    kern_size_param.trB = param().transposeB;
    kern_size_param.compute_mode = param().compute_mode;
    kern_size_param.format = param().format;

    size_t pack_size = MatrixMulForward::pack_size(param().format);
    kern_size_param.K *= pack_size;
    kern_size_param.M *= pack_size;

    return kern_size_param;
}

MatrixMulImpl::KernParam MatrixMulImpl::make_kern_param(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    KernParam kern_param;
    static_cast<KernSizeParam&>(kern_param) =
            make_kern_size_param(A.layout, B.layout, C.layout);
    kern_param.A_ptr = A.raw_ptr;
    kern_param.B_ptr = B.raw_ptr;
    kern_param.C_ptr = C.raw_ptr;
    kern_param.workspace_ptr = workspace.raw_ptr;
    kern_param.workspace_size = workspace.size;
    return kern_param;
}

size_t MatrixMulImpl::get_workspace_in_bytes(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    TensorLayoutArray layouts{A, B, C};
    HeuristicCache::Key key{this->handle(), this->get_opr_type(),
                            layouts.data(), layouts.size(),
                            &this->param(), sizeof(this->param())};
    auto rst = HeuristicCache::instance().get(key);
    if (rst.policy.algo.valid()) {
        return rst.workspace;
    }

    if (auto algo = get_algorithm_heuristic(
                A, B, C, std::numeric_limits<size_t>::max(), AlgoAttribute::DEFAULT,
                AlgoAttribute::DEFAULT)) {
        auto kern_size_param = make_kern_size_param(A, B, C);
        return static_cast<AlgoBase*>(algo)->get_workspace(kern_size_param);
    }
    return 0;
}

void MatrixMulImpl::exec(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);

    if (auto algo = get_algorithm_heuristic(
                A.layout, B.layout, C.layout, std::numeric_limits<size_t>::max(),
                AlgoAttribute::DEFAULT, AlgoAttribute::DEFAULT)) {
        auto kern_param = make_kern_param(A, B, C, workspace);
        auto kern = static_cast<AlgoBase*>(algo)->get_kern(kern_param);
        auto run = [kern, kern_param]() { kern(kern_param); };
        static_cast<naive::HandleImpl*>(handle())->dispatch_kern(run);
        return;
    }

    naive::MatrixMulForwardImpl::exec(A, B, C, workspace);
}

MatrixMulImpl::AlgoDataType MatrixMulImpl::KernSizeParam::deduce_algo_data_type()
        const {
    megdnn_assert(
            A_type.enumv() == B_type.enumv(),
            "Matmul A type and B type of different ctype\n");
    if (A_type.enumv() == DTypeEnum::Float32) {
        return MatrixMulImpl::AlgoDataType::FLOAT32;
#if !MEGDNN_DISABLE_FLOAT16
    } else if (A_type.enumv() == DTypeEnum::Float16) {
        return MatrixMulImpl::AlgoDataType::FLOAT16;
#endif
    } else if (
            A_type.enumv() == DTypeEnum::Int8 ||
            A_type.enumv() == DTypeEnum::QuantizedS8) {
        if (C_type.enumv() == DTypeEnum::Int16) {
            return MatrixMulImpl::AlgoDataType::INT8X8X16;
        } else {
            megdnn_assert(
                    C_type.enumv() == DTypeEnum::Int32 ||
                    C_type.enumv() == DTypeEnum::QuantizedS32);
            return MatrixMulImpl::AlgoDataType::QINT8X8X32;
        }
    } else if (A_type.enumv() == DTypeEnum::Quantized8Asymm) {
        return MatrixMulImpl::AlgoDataType::QUINT8X8X32;
    } else if (A_type.enumv() == DTypeEnum::Int16) {
        return MatrixMulImpl::AlgoDataType::INT16X16X32;
    } else {
        megdnn_throw(ssprintf(
                "matmul not support data type of %s * %s -> %s\n", A_type.name(),
                B_type.name(), C_type.name()));
    }
}

// vim: syntax=cpp.doxygen
