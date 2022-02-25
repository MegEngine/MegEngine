#ifndef LITE_C_COMMON_H_
#define LITE_C_COMMON_H_

#include "../src/misc.h"
#include "lite-c/network_c.h"
#include "lite-c/tensor_c.h"
#include "lite/network.h"

#if LITE_ENABLE_EXCEPTION
#include <exception>
#include <stdexcept>
#endif

//! convert c Layout to lite::Layout
lite::Layout convert_to_layout(const LiteLayout& layout);

//! convert lite::Layout to C Layout
LiteLayout convert_to_clayout(const lite::Layout& layout);

//! convert c config to lite::config
lite::Config convert_to_lite_config(const LiteConfig c_config);

//! convert C NetworkIO io to lite::NetworkIO
lite::NetworkIO convert_to_lite_io(const LiteNetworkIO c_network_io);

/*!
 * \brief handle exception
 * \param e the exception
 * \return the return value of the error
 */
int LiteHandleException(const std::exception& e);
#if LITE_ENABLE_EXCEPTION
/*! \brief  macro to guard a function */
#define LITE_CAPI_BEGIN() try {
/*! \brief every function starts with LITE_CAPI_BEGIN();
 * ends with LITE_CAPI_END or LITE_CAPI_END_WITH_STMS
 */
#define LITE_CAPI_END()                       \
    }                                         \
    catch (std::exception & _except_) {       \
        return LiteHandleException(_except_); \
    }                                         \
    return 0;
#else
/*! \brief  macro to guard a function */
#define LITE_CAPI_BEGIN() {
/*! \brief every function starts with LITE_CAPI_BEGIN();
 * ends with LITE_CAPI_END or LITE_CAPI_END_WITH_STMS
 */
#define LITE_CAPI_END() \
    }                   \
    return 0;
#endif
/*!
 * \brief catch the exception with stms
 */
#define LITE_CAPI_END_WITH_STMS(_stms)        \
    }                                         \
    catch (std::exception & _except_) {       \
        _stms;                                \
        return LiteHandleException(_except_); \
    }                                         \
    return 0;

#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
