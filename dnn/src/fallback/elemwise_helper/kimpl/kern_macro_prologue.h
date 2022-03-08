/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/kern_macro_prologue.h
 */

#define H_SWISH_KERN_FALLBACK(_func_suffix, _val1, _val2)                \
    do {                                                                 \
        auto val_zero = GiBroadcast##_func_suffix(0.f);                  \
        auto val_six = GiBroadcast##_func_suffix(6.f);                   \
        auto val_three = GiBroadcast##_func_suffix(3.f);                 \
        auto val_rec_six = GiBroadcast##_func_suffix(1.f / 6.f);         \
        auto clip1 = GiMaximum##_func_suffix(                            \
                GiMinimum##_func_suffix(                                 \
                        GiAdd##_func_suffix(_val1, val_three), val_six), \
                val_zero);                                               \
        auto clip2 = GiMaximum##_func_suffix(                            \
                GiMinimum##_func_suffix(                                 \
                        GiAdd##_func_suffix(_val2, val_three), val_six), \
                val_zero);                                               \
        _val1 = GiMultiply##_func_suffix(                                \
                GiMultiply##_func_suffix(_val1, clip1), val_rec_six);    \
        _val2 = GiMultiply##_func_suffix(                                \
                GiMultiply##_func_suffix(_val2, clip2), val_rec_six);    \
    } while (0);

#define H_SWISH_KERN_N1_FALLBACK(_func_suffix, _val1)                    \
    do {                                                                 \
        auto val_zero = GiBroadcast##_func_suffix(0.f);                  \
        auto val_six = GiBroadcast##_func_suffix(6.f);                   \
        auto val_three = GiBroadcast##_func_suffix(3.f);                 \
        auto val_rec_six = GiBroadcast##_func_suffix(1.f / 6.f);         \
        auto clip1 = GiMaximum##_func_suffix(                            \
                GiMinimum##_func_suffix(                                 \
                        GiAdd##_func_suffix(_val1, val_three), val_six), \
                val_zero);                                               \
        _val1 = GiMultiply##_func_suffix(                                \
                GiMultiply##_func_suffix(_val1, clip1), val_rec_six);    \
    } while (0);

// vim: syntax=cpp.doxygen
