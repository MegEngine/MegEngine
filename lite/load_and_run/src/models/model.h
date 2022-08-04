#pragma once
#include <gflags/gflags.h>
#include <string>
#include "helpers/common.h"
#include "megbrain/utils/json.h"
DECLARE_bool(lite);
DECLARE_bool(mdl);

namespace lar {
/*!
 * \brief: base class of model
 */
class ModelBase {
public:
    //! get model type by the magic number in dump file
    static ModelType get_model_type(std::string model_path);

    //! create model by different model type
    static std::shared_ptr<ModelBase> create_model(std::string model_path);

    //! type of the model
    virtual ModelType type() = 0;

    //! set model load state

    virtual void set_shared_mem(bool state) = 0;

    virtual void create_network(){};

    //! load model interface for load and run strategy
    virtual void load_model() = 0;

    //! run model interface for load and run strategy
    virtual void run_model() = 0;

    //! wait asynchronous function interface for load and run strategy
    virtual void wait() = 0;

    virtual ~ModelBase() = default;

    virtual const std::string& get_model_path() const = 0;

    virtual std::vector<uint8_t> get_model_data() = 0;
#if MGB_ENABLE_JSON
    //! get model io information
    virtual std::shared_ptr<mgb::json::Object> get_io_info() = 0;
#endif
};
}  // namespace lar

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
