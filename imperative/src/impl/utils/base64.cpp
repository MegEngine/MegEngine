/**
 * \file imperative/src/impl/base64.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/utils/base64.h"

namespace mgb::imperative {

namespace {

/*
** Translation Table as described in RFC1113
*/
const char cb64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/*
** Translation Table to decode:
*https://github.com/dgiardini/imgcalkap/blob/master/base64.c
*/
const char cd64[] =
        "|$$$}rstuvwxyz{$$$$$$$>?@ABCDEFGHIJKLMNOPQRSTUVW$$$$$$XYZ[\\]^_`"
        "abcdefghijklmnopq";

/*
** encodeblock
**
** encode 3 8-bit binary bytes as 4 '6-bit' characters
*/
void encodeblock(unsigned char in[3], unsigned char out[4], int len) {
    out[0] = cb64[in[0] >> 2];
    out[1] = cb64[((in[0] & 0x03) << 4) | ((in[1] & 0xf0) >> 4)];
    out[2] =
            (unsigned char)(len > 1 ? cb64[((in[1] & 0x0f) << 2) | ((in[2] & 0xc0) >> 6)] : '=');
    out[3] = (unsigned char)(len > 2 ? cb64[in[2] & 0x3f] : '=');
}

/*
** decodeblock
**
** decode 4 '6-bit' characters into 3 8-bit binary bytes
*/
void decodeblock(unsigned char in[4], unsigned char out[3]) {
    out[0] = (unsigned char)(in[0] << 2 | in[1] >> 4);
    out[1] = (unsigned char)(in[1] << 4 | in[2] >> 2);
    out[2] = (unsigned char)(((in[2] << 6) & 0xc0) | in[3]);
}

}  // namespace

/**
 * Encode string to base64 string
 * @param input - source string
 * @param outdata - target base64 string
 * @param linesize - max size of line
 */
void encode(
        const std::vector<std::uint8_t>& input, std::vector<std::uint8_t>& outdata,
        int linesize) {
    outdata.clear();

    unsigned char in[3], out[4];
    int i, len, blocksout = 0;
    size_t j = 0;

    auto* indata = reinterpret_cast<const unsigned char*>(input.data());
    unsigned int insize = input.size();

    while (j <= insize) {
        len = 0;
        for (i = 0; i < 3; i++) {
            in[i] = (unsigned char)indata[j];
            j++;
            if (j <= insize) {
                len++;
            } else {
                in[i] = 0;
            }
        }
        if (len) {
            encodeblock(in, out, len);
            for (i = 0; i < 4; i++) {
                outdata.push_back(out[i]);
            }
            blocksout++;
        }
        if (blocksout >= (linesize / 4) || (j == insize)) {
            if (blocksout) {
                outdata.push_back('\r');
                outdata.push_back('\n');
            }
            blocksout = 0;
        }
    }
}

/**
 * Decode base64 string ot source
 * @param input - base64 string
 * @param outdata - source string
 */
void decode(
        const std::vector<std::uint8_t>& input, std::vector<std::uint8_t>& outdata) {
    outdata.clear();

    unsigned char in[4], out[3], v;
    int i, len;
    size_t j = 0;

    auto* indata = reinterpret_cast<const unsigned char*>(input.data());
    unsigned int insize = input.size();

    while (j <= insize) {
        for (len = 0, i = 0; i < 4 && (j <= insize); i++) {
            v = 0;
            while ((j <= insize) && v == 0) {
                v = (unsigned char)indata[j++];
                v = (unsigned char)((v < 43 || v > 122) ? 0 : cd64[v - 43]);
                if (v) {
                    v = (unsigned char)((v == '$') ? 0 : v - 61);
                }
            }
            if (j <= insize) {
                len++;
                if (v) {
                    in[i] = (unsigned char)(v - 1);
                }
            } else {
                in[i] = 0;
            }
        }
        if (len) {
            decodeblock(in, out);
            for (i = 0; i < len - 1; i++) {
                outdata.push_back(out[i]);
            }
        }
    }
}

/**
 * Encode binary data to base64 buffer
 * @param input - source data
 * @param outdata - target base64 buffer
 * @param linesize
 */
void encode(const std::string& input, std::string& outdata, int linesize) {
    std::vector<std::uint8_t> out;
    std::vector<std::uint8_t> in(input.begin(), input.end());
    encode(in, out, linesize);
    outdata = std::string(out.begin(), out.end());
}

/**
 * Decode base64 buffer to source binary data
 * @param input - base64 buffer
 * @param outdata - source binary data
 */
void decode(const std::string& input, std::string& outdata) {
    std::vector<std::uint8_t> in(input.begin(), input.end());
    std::vector<std::uint8_t> out;
    decode(in, out);
    outdata = std::string(out.begin(), out.end());
}

}  // namespace mgb::imperative
