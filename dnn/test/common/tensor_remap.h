#pragma once
#include "megdnn/oprs.h"

#include "test/common/index.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {
namespace tensor_remap {

class MapRNG final : public IIDRNG {
public:
    MapRNG(TensorShape src) : m_cnt(0), m_src(src) {}
    dt_float32 gen_single_val() override;

private:
    size_t m_cnt;
    TensorShape m_src;
};

class NonoverlappingMapRNG final : public IIDRNG {
public:
    NonoverlappingMapRNG(TensorShape src);
    dt_float32 gen_single_val() override;

private:
    size_t m_cnt;
    TensorShape m_src;
    Index m_idx;
};

}  // namespace tensor_remap
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
