/**
 * \file src/gopt/include/megbrain/gopt/reformat_emitter.h
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
#include <vector>
#include "megbrain/graph.h"
#include "megdnn/named_tensor.h"

namespace mgb {
namespace gopt {

class Emitter {
public:
    using Builder = thin_function<VarNode*(const VarNodeArray&)>;
    using Checker = thin_function<bool(const VarNodeArray&)>;
    using EmitResult = std::tuple<Builder, Checker>;
    virtual ~Emitter() = default;
    virtual EmitResult emit() const = 0;
};

class ModifyShapeMixin {
protected:
    using Pattern = SmallVector<std::tuple<int, int, bool>>;
    using Checker = Emitter::Checker;
    ModifyShapeMixin(const megdnn::NamedTensorShape& src,
                     const megdnn::NamedTensorShape& dest)
            : m_src(src), m_dest(dest) {}
    Pattern mixin_analyze() const;
    Checker mixin_emit_checker(const Pattern& pattern) const;
    megdnn::NamedTensorShape m_src, m_dest;
};

class MakeShapeEmitter final : public Emitter, ModifyShapeMixin {
public:
    MakeShapeEmitter(const megdnn::NamedTensorShape& src,
                     const megdnn::NamedTensorShape& dest)
            : ModifyShapeMixin(src, dest) {}
    EmitResult emit() const override;
};

class ReshapeEmitter final : public Emitter, ModifyShapeMixin {
public:
    ReshapeEmitter(const megdnn::NamedTensorShape& src,
                   const megdnn::NamedTensorShape& dest)
            : ModifyShapeMixin(src, dest) {}
    EmitResult emit() const override;
};

class DimshuffleEmitter final : public Emitter {
public:
    DimshuffleEmitter(const std::vector<int>& pattern) : m_pattern{pattern} {}
    EmitResult emit() const override;

private:
    std::vector<int> m_pattern;
};

class ReformatEmitter final : public Emitter, ModifyShapeMixin {
public:
    ReformatEmitter(const megdnn::NamedTensorShape& src,
                    const megdnn::NamedTensorShape& dest)
            : ModifyShapeMixin(src, dest) {}
    EmitResult emit() const override;

private:
    struct UnderlyingBuilders {
        Builder make_shape1, make_shape2, reshape1, reshape2, dimshuffle;
    };
    UnderlyingBuilders analyze() const;
};
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
