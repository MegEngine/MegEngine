#if MGB_ENABLE_FBS_SERIALIZATION

#include "megbrain/gopt/framework.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/version.h"

namespace mgb {
namespace serialization {

constexpr uint32_t MGB_VERSION = (MGE_MAJOR * 1000 + MGE_MINOR) * 100 + MGE_PATCH;

constexpr uint32_t MGB_MAGIC = 0x4342474D;

// In order to maintain compatibility and to allow old models to be loaded, we keep
// the old magic(MAGIC_V0) value and creat a new magic(MGB_MAGIC)
constexpr uint32_t MAGIC_V0 = 0x5342474D;

void check_tensor_value_valid(const std::string& name, const HostTensorND& tensor);

template <typename T>
bool contains_any_in_set(const SmallVector<T>& list, const ThinHashSet<T>& set) {
    for (const auto& x : list) {
        if (set.count(x)) {
            return true;
        }
    }
    return false;
}

/*!
 * \brief replace the the opr who has the replace_opr methord in OprLoadDumpImplV2
 */
template <class T>
class PassConvertToCompatible : public gopt::Pass {
    ThinHashMap<
            Typeinfo*, thin_function<cg::OperatorNodeBase*(
                               cg::OperatorNodeBase*, const VarNodeArray&)>>
            m_opr_replace_func;
    gopt::VarReplaceCheckFlag m_var_replace_check_flag =
            gopt::VarReplaceCheckFlag::CHECK_ALL;

public:
    const char* name() const override { return "PassConvertToCompatible"; };

    PassConvertToCompatible& set_var_replace_check_flag(
            gopt::VarReplaceCheckFlag flag) {
        m_var_replace_check_flag = flag;
        return *this;
    }

    void apply(gopt::OptState& state) const override {
        state.set_var_replace_check_flag(m_var_replace_check_flag);
        auto rewriter = state.graph().make_rewriter();

        auto on_opr = [this, &rewriter](cg::OperatorNodeBase* opr) {
            auto it = m_opr_replace_func.find(opr->dyn_typeinfo());
            if (it != m_opr_replace_func.end()) {
                VarNodeArray new_inp;
                new_inp.clear();
                new_inp.reserve(opr->input().size());
                for (auto i : opr->input()) {
                    new_inp.push_back(rewriter.get_var(i));
                }
                auto new_opr = (it->second)(opr, new_inp);

                auto &&origin_out = opr->output(), &&cur_out = new_opr->output();
                if (opr == new_opr) {
                    rewriter.auto_replace_outputs(opr);
                } else {
                    for (size_t i = 0; i < std::min(origin_out.size(), cur_out.size());
                         i++) {
                        rewriter.replace_var(origin_out[i], cur_out[i], nullptr);
                    }
                }
            } else {
                rewriter.auto_replace_outputs(opr);
            }
        };
        state.graph().iter(on_opr);
        rewriter.apply_inplace();
    }

    static std::unique_ptr<PassConvertToCompatible> make(
            const SymbolVarArray& output_vars,
            std::function<const void*(cg::OperatorNodeBase*)> get_reg) {
        auto ret = std::make_unique<PassConvertToCompatible>();
        // iterate oprs to init
        auto on_opr = [&](cg::OperatorNodeBase* opr) {
            if (!GraphDumper::should_remove_in_dump(opr)) {
                auto registry = reinterpret_cast<const T*>(get_reg(opr));
                mgb_throw_if(
                        !registry,
                        cg::OperatorNodeExcExtraInfo::ExcMaker{opr}.make<MegBrainError>,
                        "serialization as FlatBuffers is not supported for "
                        "operator %s, typeinfo %p",
                        opr->dyn_typeinfo()->name, opr->dyn_typeinfo());
                if (registry->converter) {
                    ret->m_opr_replace_func[opr->dyn_typeinfo()] = registry->converter;
                }
            }
        };
        cg::DepOprIter dep_opr_iter{on_opr};
        for (auto i : output_vars) {
            dep_opr_iter.add(i.node()->owner_opr());
        }
        return ret;
    };
};

}  // namespace serialization
}  // namespace mgb

#endif
