#include "megbrain/graph.h"
#include "megbrain_build_config.h"

namespace mgb {
namespace opr {
namespace standalone {

/*!
 * \brief generate indices of boxes to be kept after NMS
 *
 * See the docs in the python operator
 */
MGB_DEFINE_OPR_CLASS(NMSKeep,
                     cg::SingleCNOutshapePureByInshapeOprBase) // {
public:
    struct Param {
        //! TAG is used by the serializer to check Param type; here we
        //! just use a random number. To generate such a random number,
        //! run `xxd -l4 -p /dev/urandom`
        static constexpr uint32_t TAG = 0x988a7630u;

        float iou_thresh;     //!< IoU threshold for overlapping
        uint32_t max_output;  //!< max number of output boxes per batch
    };
    

    NMSKeep(VarNode * boxes, const Param& param,
            const OperatorNodeConfig& config);
    ~NMSKeep() noexcept;

    //! factory method to insert the operator into a graph
    static SymbolVar make(SymbolVar boxes, const Param& param,
                          const OperatorNodeConfig& config = {});

    const Param& param() const { return m_param; }

private:
    const Param m_param;

    class Kern;
    class CUDAKern;
    class CPUKern;

    std::unique_ptr<Kern> m_kern;

    //! override output shape infer func provided by
    //! SingleCNOutshapePureByInshapeOprBase
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;

    //! this opr requires inputs to be contiguous
    void add_input_layout_constraint() override;

    //! execute the operator
    void scn_do_execute() override;
};

}  // namespace standalone
}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
