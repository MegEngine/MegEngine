/**
 * \file src/gopt/include/megbrain/gopt/reformat_manager.h
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
#include "megbrain/gopt/reformat_emitter.h"
#include "megbrain/graph.h"

namespace mgb {
namespace gopt {

enum class TensorType : uint32_t {
    FEATURE = 0,
    WEIGHT = 1,
};

enum class TensorFormats : uint32_t {
    // input tensor formats
    NCHW = 0,     ///< [N, C, H, W]
    NHWC = 1,     ///< [N, H, W, C]
    NCHWc4 = 2,   ///< [N, C/4, H, W, C%4]
    NCHWc8 = 3,   ///< [N, C/8, H, W, C%8]
    NCHWc32 = 4,  ///< [N, C/32, H, W, C%32]
    NCHWc64 = 5,  ///< [N, C/64, H, W, C%64]
    CHWNc4 = 6,   ///< [C/4, H, W, N, C%4]
    NHCWc4 = 7,   ///< [N, H, C/4, W, C%4]
    // weight tensor formats
    // NHWCD4
    KRSCk4 = 8,   ///< [K/4, R, S, C, K%4] [dense conv]
    GKRSCk4 = 9,  ///< [G, K/4, R, S, C, K%4] [group conv]
    C1RSc4 = 10,  ///< [C/4, 1, R, S, C%4] [channel wise conv]
    // NHWCD4-dot
    KRSCk4c4 = 11,   ///< [K/4, R, S, C/4, K%4, C%4] [dense conv]
    GKRSCk4c4 = 12,  ///< [G, K/4, R, S, C/4, K%4, C%4] [group conv]
    // NCHW44-dot
    KCRSk4c4 = 13,   ///< [K/4, C/4, R, S, K%4, C%4] [dense conv]
    GKCRSk4c4 = 14,  ///< [G, K/4, C/4, R, S, K%4, C%4] [group conv]
    // NCHW44
    KCRSc4k4 = 15,   ///< [K/4, C/4, R, S, C%4, K%4] [dense conv]
    GKCRSc4k4 = 16,  ///< [G, K/4, C/4, R, S, C%4, K%4] [group conv]
    C11RSc4 = 17,    ///< [C/4, 1, 1, R, S, C%4] [channel wise conv]
    // NCHW88
    KCRSc8k8 = 18,   ///< [K/8, C/8, R, S, C%8, K%8] [dense conv]
    GKCRSc8k8 = 19,  ///< [G, K/8, C/8, R, S, C%8, K%8] [group conv]
    C11RSc8 = 20,    ///< [C/8, 1, 1, R, S, C%8] [channel wise conv]

    KRSCk8 = 21,  ///< [K/8, R, S, C, K%8]

    // NCHW4
    KCRSc4 = 22,   ///< [K, C/4, R, S, C%4]
    GKCRSc4 = 23,  ///< [G, K, C/4, R, S, C%4]

    // default weight format
    KCRS = 24,   ///< [K, C, R, S]
    GKCRS = 25,  ///< [G, K, C, R, S]
    C11RS = 26,  ///< [C, 1, 1, R, S]
};

class ReformatManager : public NonCopyableObj {
    ReformatManager();

public:
    using ReformatImpl = thin_function<VarNode*(const VarNodeArray&)>;
    struct ReformatKey {
        enum class Attribute : uint32_t {
            DEFAULT = 0,
            IMAGE2D = 1 << 0,
            IC_SMALL = 1 << 1,
        };
        TensorFormats input_format, output_format;
        DTypeEnum input_dtype, output_dtype;
        Attribute attribute;
        std::string to_string() const;
        ReformatKey()
                : input_dtype{DTypeEnum::Float32},
                  output_dtype{DTypeEnum::Float32},
                  attribute{Attribute::DEFAULT} {}
        ReformatKey(TensorFormats input_format_, TensorFormats output_format_,
                    Attribute attribute_ = Attribute::DEFAULT,
                    DTypeEnum input_dtype_ = DTypeEnum::Float32,
                    DTypeEnum output_dtype_ = DTypeEnum::Float32)
                : input_format{input_format_},
                  output_format{output_format_},
                  input_dtype{input_dtype_},
                  output_dtype{output_dtype_},
                  attribute{attribute_} {}
        struct Hash {
            size_t operator()(const ReformatKey& key) const;
        };
        struct Equal {
            bool operator()(const ReformatKey& lhs,
                            const ReformatKey& rhs) const;
        };
        ReformatKey& deduce_reformat_dtype_enum(const DType& dt);
    };
    using ReformatCache =
            std::unordered_map<ReformatKey, ReformatImpl, ReformatKey::Hash,
                               ReformatKey::Equal>;
    ReformatImpl get(const ReformatKey& key) const;
    ReformatImpl get(ReformatKey&& key) const { return get(key); }
    ReformatImpl auto_aligned_reformat_featrue(const VarNode* orig_var,
                                               TensorFormats orig_format,
                                               const ReformatKey& key) const;
    struct AlignmentDesc {
        megdnn::Dimension::Name name;
        size_t alignment;
    };
    ReformatImpl auto_aligned_reformat_weight(
            const VarNode* orig_var, const ReformatKey& key,
            const AlignmentDesc& extra_alignment = {}) const;
    static const ReformatManager& instance();

private:
    ReformatCache m_cache;
};

TensorShape make_aligned_tensor_shape(const VarNode* var,
                                      TensorFormats orig_formats,
                                      TensorFormats target_formats);

TensorShape make_aligned_weight_shape(const VarNode* var,
                                      TensorFormats orig_formats,
                                      TensorFormats target_formats,
                                      TensorFormats extra_formats);
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
