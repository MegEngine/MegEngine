#pragma once

#include "megbrain/common.h"

namespace mgb::imperative {

/**
 * Encode string to base64 string
 * @param input - source string
 * @param outdata - target base64 string
 * @param linesize - max size of line
 */
void encode(
        const std::vector<std::uint8_t>& input, std::vector<std::uint8_t>& outdata,
        int linesize = 76);

/**
 * Decode base64 string ot source
 * @param input - base64 string
 * @param outdata - source string
 */
void decode(const std::vector<std::uint8_t>& input, std::vector<std::uint8_t>& outdata);

/**
 * Encode binary data to base64 buffer
 * @param input - source data
 * @param outdata - target base64 buffer
 * @param linesize
 */
void encode(const std::string& input, std::string& outdata, int linesize = 76);

/**
 * Decode base64 buffer to source binary data
 * @param input - base64 buffer
 * @param outdata - source binary data
 */
void decode(const std::string& input, std::string& outdata);

}  // namespace mgb::imperative
