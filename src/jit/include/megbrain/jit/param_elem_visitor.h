#pragma once

/*
 * please note that all arithmetics on GPU are 32-bit for best performance;
 * this
 * limits max possible size
 */

/*!
 * \brief fast division for unsigned int
 */
struct Uint32Fastdiv {
    unsigned int m_mul, m_divisor, m_divisor_is_not_1, m_inc_dividend, m_shift;

    static const unsigned int MAX_DIVIDEND = ~0u - 1;
};

template <int ndim>
struct ParamElemVisitor {
    int m_stride[ndim];

    //! m_shape_highdim[i] = original_shape[i + 1]
    Uint32Fastdiv m_shape_highdim[ndim > 1 ? ndim - 1 : 1];
    static const int NDIM = ndim;
};

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
