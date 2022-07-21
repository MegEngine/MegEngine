#include "./base.cuh"

using namespace convolution;

Uint32Fastdiv::Uint32Fastdiv() {
    memset(this, 0, sizeof(Uint32Fastdiv));
}

Uint32Fastdiv& Uint32Fastdiv::operator=(uint32_t d) {
    m_divisor = d;
    constexpr uint32_t MAX_U32 = ~0u;
    m_inc_dividend = 0;
    m_divisor_is_not_1 = ~0u;
    if (!(d & (d - 1))) {
        // power of 2
        m_mul = 1u << 31;
        int p = 0;
        while ((1u << p) < d)
            ++p;
        m_shift = p ? p - 1 : 0;
        if (d == 1)
            m_divisor_is_not_1 = 0;
        return *this;
    }
    auto n_bound = uint64_t(d / 2 + 1) * MAX_U32;
    uint32_t shift = 32;
    while ((1ull << shift) < n_bound)
        ++shift;
    uint64_t mdst = 1ull << shift;
    int64_t delta = d - mdst % d;
    m_mul = mdst / d + 1;
    if ((uint64_t)delta > d / 2) {
        delta -= d;
        --m_mul;
        m_inc_dividend = 1;
    }
    m_shift = shift - 32;
    return *this;
}
