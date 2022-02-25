#include "./opr_safe_dump.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_manip.h"

#include "megbrain/utils/hash_ct.h"
#include "midout.h"
MIDOUT_DECL(megbrain_opr_safe_dump)
#define MIDOUT_B(...) MIDOUT_BEGIN(megbrain_opr_safe_dump, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace opr;

namespace {
template <typename Param>
void write_param(std::string& data, const Param& param) {
    megdnn::Algorithm::serialize_write_pod(param, data);
}

template <>
void write_param(std::string& /* data */, const DType& /* dtype */) {}

template <class Opr>
struct OprDumpImpl;

#define cb(_Opr)                                                    \
    template <>                                                     \
    struct OprDumpImpl<_Opr> {                                      \
        static std::string dump(const cg::OperatorNodeBase* opr_) { \
            MIDOUT_B(_Opr)                                          \
            auto&& opr = opr_->cast_final_safe<_Opr>();             \
            std::string data;                                       \
            auto opr_hash = MGB_HASH_STR(#_Opr);                    \
            write_param(data, opr_hash);                            \
            write_param(data, opr.param());                         \
            return data;                                            \
            MIDOUT_E                                                \
        }                                                           \
    };
FOREACH_SUPPORTED_OPR_WITHOUT_EXECUTION_POLICY(cb)
#undef cb

#define cb(_Opr)                                                       \
    template <>                                                        \
    struct OprDumpImpl<_Opr> {                                         \
        static std::string dump(const cg::OperatorNodeBase* opr_) {    \
            MIDOUT_B(_Opr)                                             \
            auto&& opr = opr_->cast_final_safe<_Opr>();                \
            std::string data;                                          \
            auto opr_hash = MGB_HASH_STR(#_Opr);                       \
            write_param(data, opr_hash);                               \
            write_param(data, opr.param());                            \
            using ExecutionPolicy = megdnn::param::ExecutionPolicy;    \
            ExecutionPolicy policy{                                    \
                    opr.execution_policy_transient().strategy,         \
                    opr.execution_policy_transient().workspace_limit}; \
            write_param(data, policy);                                 \
            return data;                                               \
            MIDOUT_E                                                   \
        }                                                              \
    };
FOREACH_SUPPORTED_OPR_WITH_EXECUTION_POLICY(cb)
#undef cb
}  // namespace

namespace mgb {
namespace gopt {
namespace intl {

std::string opr_safe_dump(const cg::OperatorNodeBase* opr) {
#define cb(_Opr)                                   \
    if (opr->dyn_typeinfo() == _Opr::typeinfo()) { \
        return OprDumpImpl<_Opr>::dump(opr);       \
    } else
    FOREACH_SUPPORTED_OPR(cb) {
        mgb_throw(
                InternalError, "unsupported operator(got:%s)",
                opr->dyn_typeinfo()->name);
    }
#undef cb
}

}  // namespace intl
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
