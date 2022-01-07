#pragma once

#include "megbrain/tensor.h"

namespace mgb::imperative {

/**
 * \brief like TensorFormats, but only including common formats and DEFAULT.
 *
 */
class Format {
public:
    enum class Type {
        DEFAULT = 0,
        NCHW = 1,  ///< [N, C, H, W]
        NHWC = 2,  ///< [N, H, W, C]
    };
    std::string to_string() const {
        switch (m_type) {
            case Type::DEFAULT:
                return "default";
            case Type::NCHW:
                return "nchw";
            case Type::NHWC:
                return "nhwc";
            default:
                mgb_throw(MegBrainError, "bad format type");
        }
    }
    Format() : m_type(Type::DEFAULT) {}
    Format(std::string str) {
        if (str == "default") {
            m_type = Type::DEFAULT;
        } else if (str == "nchw") {
            m_type = Type::NCHW;
        } else if (str == "nhwc") {
            m_type = Type::NHWC;
        } else {
            mgb_throw(
                    MegBrainError,
                    "Invalid format type."
                    " Only support \"default\", \"nchw\" and \"nhwc\"");
        }
    }
    Format(Type type) : m_type(type) {}
    Type type() const { return m_type; }
    bool operator==(const Format& b) const { return m_type == b.type(); }
    bool operator==(const Format::Type& b) const { return m_type == b; }
    bool operator!=(const Format& b) const { return m_type != b.type(); }
    bool operator!=(const Format::Type& b) const { return m_type != b; }

private:
    Type m_type = Type::DEFAULT;
};

}  // namespace mgb::imperative
