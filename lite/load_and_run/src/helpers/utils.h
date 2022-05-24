#pragma once

#include <vector>
#include "common.h"
#include "json_loader.h"
#include "megbrain/utils/json.h"
namespace lar {
/**
 * fitting profiler type
 */
enum class ProiflerType {
    TIME_PROFILER = 0,
    UNSPEC_PROFILER = 1,
};
/**
 * option coder type
 */
enum class CoderType {
    GFLAGS = 0,
    JSON = 1,
    UNSPEC = 2,
};

/**
 *  option coder to transform internal option val into differnet form
 */
class OptionsCoder {
public:
    OptionsCoder(){};
    //! encode options into given format
    virtual std::string encode(OptionValMap&, bool) = 0;

    //! decode options with given format into option map
    virtual OptionValMap& decode(const std::string&, OptionValMap& val_map) = 0;

    //! destructor
    virtual ~OptionsCoder() = default;
};

#if MGB_ENABLE_JSON
class JsonOptionsCoder final : public OptionsCoder {
public:
    JsonOptionsCoder(){};

    //! encode given options into json format
    std::string encode(OptionValMap&, bool encode_all) override;

    std::vector<std::shared_ptr<mgb::json::Object>> encode(OptionValMap&);

    //! decode given json format options into given options map
    OptionValMap& decode(const std::string&, OptionValMap&) override;

private:
    mgb::JsonLoader m_json_loader;
};
#endif

class GflagsOptionsCoder final : public OptionsCoder {
public:
    GflagsOptionsCoder(){};

    //! encode given options into gflags format
    std::string encode(OptionValMap&, bool encode_all = false) override;

    //! decode given gflags format options into given options maps
    OptionValMap& decode(const std::string&, OptionValMap&) override;
};

}  // namespace lar