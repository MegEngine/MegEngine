/**
 * \file sdk/load-and-run/test/test_json_loader.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "../src/json_loader.h"

using namespace mgb;

void test_number(double real, std::string str) {
    JsonLoader json;
    auto root = json.load(str.data(), str.size());
    mgb_assert(root->is_number());
    mgb_assert(std::fabs(real - root->number()) <= DBL_EPSILON);
}

void test_string(std::string str, std::string json_str) {
    JsonLoader json;
    auto root = json.load(json_str.data(), json_str.size());
    mgb_assert(root->is_str());
    mgb_assert(str == root->str());
}

void test_array(size_t num, std::string str) {
    JsonLoader json;
    auto root = json.load(str.data(), str.size());
    mgb_assert(root->is_array());
    mgb_assert(root->len() == num);
}

void test_object(size_t num, std::string str) {
    JsonLoader json;
    auto root = json.load(str.data(), str.size());
    mgb_assert(root->is_object());
    mgb_assert(root->len() == num);
}

int main() {
    test_number(1.0, "1.0");
    test_number(1e10, "1e10");
    test_number(0.2345678, "0.02345678e1");
    test_number(-10086, "-1.0086E4");
    test_number(1.7976931348623157e+308,
                "1.7976931348623157e+308");  // max double

    test_string("a", "\"a\"");
    test_string("\\table", "\"\\table\"");

    test_array(0, "  [  ] ");
    test_array(4, " [  0.1,  0.2,0.3, 1990 ] ");
    test_array(2, " [  0.1,  \"hello-world\"]");
    test_array(3, " [  0.1,  \"hello-world\", [2.0, 33]]");
    test_array(1, " [ [  [ [2020] ], [2021], [[2022]]  ] ]");

    test_object(0, "   {      }            ");
    test_object(1, "{\"key1\": 2023}");
    test_object(1,
                "{\"key1\": { 		\"key2\": { 			"
                "\"key3\": \"value\" 		} 	} }");
    test_object(1, "{\"key1\":{\"key2\":{}}}");

    printf("test passed\n");
    return 0;
}