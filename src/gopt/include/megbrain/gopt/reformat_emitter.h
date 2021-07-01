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
    using Operator = thin_function<VarNode*(VarNode*)>;
    virtual ~Emitter() = default;
    virtual Operator emit() const = 0;
};

class ReshapeEmitter final : public Emitter {
public:
    using Operator = typename Emitter::Operator;
    ReshapeEmitter(const megdnn::NamedTensorShape& src,
                   const megdnn::NamedTensorShape& dest)
            : m_src{src}, m_dest{dest} {}
    Operator emit() const override;

private:
    SmallVector<std::tuple<int, int, bool>> analyze() const;
    megdnn::NamedTensorShape m_src, m_dest;
};

class DimshuffleEmitter final : public Emitter {
public:
    using Operator = typename Emitter::Operator;
    DimshuffleEmitter(const std::vector<int>& pattern) : m_pattern{pattern} {}
    Operator emit() const override;

private:
    std::vector<int> m_pattern;
};

class ReformatEmitter final : public Emitter {
public:
    using Operator = typename Emitter::Operator;
    ReformatEmitter(const megdnn::NamedTensorShape& src,
                    const megdnn::NamedTensorShape& dest)
            : m_src{src}, m_dest{dest} {}
    Operator emit() const override;

private:
    SmallVector<Operator> analyze() const;
    megdnn::NamedTensorShape m_src, m_dest;
};
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
