#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "megbrain/opr/io.h"

namespace lar {
/*!
 * \brief data parser for --input
 * support .json|.ppm|.pgm|.npy data and user define data string
 * data string format: [0,0,227,227]
 */
struct DataParser {
    struct Brace {
        std::weak_ptr<Brace> parent;
        std::vector<std::shared_ptr<Brace>> chidren;
    };
    void feed(const std::string& path);

    std::unordered_map<std::string, mgb::HostTensorND> inputs;

private:
    //! parser for json data
    void parse_json(const std::string& path);

    //! parser for .ppm .pgm image
    void parse_image(const std::string& name, const std::string& path);

    //! parser for .npy data
    void parse_npy(const std::string& name, const std::string& path);

    //! parser for user define string
    void parse_string(const std::string name, const std::string& str);
};
}  // namespace lar
