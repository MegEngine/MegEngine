#include "megbrain/gopt/profiler.h"
#include "megbrain/gopt/solver.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"

using namespace mgb;
using namespace gopt;
using namespace opr;

namespace {
using OprFormat = SolverBase::OprFormat;
template <typename Opr>
bool check_format_aware_opr_valid(const OperatorNodeBase* opr_, OprFormat opr_format) {
    auto&& opr = opr_->cast_final_safe<Opr>();
    return opr.param().format == opr_format;
}
}  // namespace

/* =================== ProfilingBasedSolverSolver ======================*/
ProfilingBasedSolver::ProfilingBasedSolver(std::unique_ptr<ProfilerBase> profiler)
        : m_profiler{std::move(profiler)} {
    static const ThinHashMap<
            Typeinfo*,
            thin_function<bool(const OperatorNodeBase*, OprFormat opr_format)>>
            format_aware_opr_validators = {
#define cb(t)                                                          \
    {opr::t::typeinfo(), std::bind(                                    \
                                 check_format_aware_opr_valid<opr::t>, \
                                 std::placeholders::_1, std::placeholders::_2)}
                    cb(Convolution),
                    cb(ConvBiasForward),
                    cb(ConvolutionBackwardData),
                    cb(PoolingForward),
                    cb(WarpPerspective),
                    cb(Resize),
            };

    m_problem_filter = [](const Problem& problem) {
        auto&& base_opr_format = OprTensorFormatsConfiguration::safe_cast_to_opr_format(
                problem.attribute().base_config_id);
        bool has_format_aware_opr = false;
        for (auto&& opr : problem.graph_partition().all_oprs()) {
            auto iter = format_aware_opr_validators.find(opr->dyn_typeinfo());
            if (iter != format_aware_opr_validators.end() &&
                iter->second(opr, base_opr_format)) {
                has_format_aware_opr = true;
                break;
            }
        }
        return has_format_aware_opr;
    };
}

ProfilingBasedSolver::Solution ProfilingBasedSolver::solve(
        const Problem& problem) const {
    if (!m_problem_filter(problem))
        return Solution{};
    return do_solve(problem);
}

// vim: syntax=cpp.doxygen
