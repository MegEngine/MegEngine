/**
 * \file src/jit/impl/mlir/ir/numerical.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "numerical.h"

namespace mgb {
namespace jit {

mlir::Value polynomial(ValueBuilderHelper& helper, mlir::Value x,
                       std::vector<mlir::Value>& coeff) {
    size_t n = coeff.size();
    if (n == 0) {
        return helper.const_f32(0);
    }

    mlir::Value r = coeff[0];
    for (size_t i = 1; i < n; i++) {
        r = helper.add(helper.mul(r, x), coeff[i]);
    }
    return r;
}

// polynomial approximation of arctangent
// atan(t) = t + c3 * t^3 + c5 * t^5 + ... + c17 * t^17
// original paper:
// https://arxiv.org/pdf/1508.03211.pdf
mlir::Value atan2_approx(ValueBuilderHelper& helper, mlir::Value y,
                         mlir::Value x) {
    auto atan_poly = [&](mlir::Value t) {
        std::vector<mlir::Value> coeff = {
                helper.const_f32(2.90188402868807315826416015625E-3),
                helper.const_f32(-1.62907354533672332763671875E-2),
                helper.const_f32(4.3082617223262786865234375E-2),
                helper.const_f32(-7.5408883392810821533203125E-2),
                helper.const_f32(0.1066047251224517822265625),
                helper.const_f32(-0.14209578931331634521484375),
                helper.const_f32(0.19993579387664794921875),
                helper.const_f32(-0.3333314359188079833984375)};
        auto t2 = helper.mul(t, t);
        auto p = polynomial(helper, t2, coeff);
        return helper.add(helper.mul(helper.mul(p, t2), t), t);
    };

    // constants
    auto zero = helper.const_f32(0);
    auto pi = helper.const_f32(3.141592653589793);
    auto pi_over_2 = helper.const_f32(1.570796326794897);

    // transform the angle into interval [0, pi/4]
    auto ax = helper.abs(x);
    auto ay = helper.abs(y);
    auto q = helper.div(helper.min(ax, ay), helper.max(ax, ay));

    // get approximation for interval [0, pi/4]
    auto r = atan_poly(q);

    // [0, pi/4] => [0, pi/2]
    r = helper.select(helper.le(ax, ay), helper.sub(pi_over_2, r), r);

    // [0, pi/2] => [0, pi]
    r = helper.select(helper.le(x, zero), helper.sub(pi, r), r);

    // [0, pi] => [-pi, pi]
    r = helper.select(helper.le(y, zero), helper.sub(zero, r), r);

    return r;
}

// numerical approximation of gauss error function
// https://en.wikipedia.org/wiki/Error_function#Polynomial
// original book:
// Numerical Recipes in Fortran 77: The Art of Scientific Computing
mlir::Value erf_approx(ValueBuilderHelper& helper, mlir::Value x) {
    auto zero = helper.const_f32(0);
    auto one = helper.const_f32(1);
    auto half = helper.const_f32(0.5);

    auto t = helper.div(one, helper.add(one, helper.mul(half, helper.abs(x))));

    std::vector<mlir::Value> coeff = {
            helper.const_f32(0.17087277),
            helper.const_f32(-0.82215223),
            helper.const_f32(1.48851587),
            helper.const_f32(-1.13520398),
            helper.const_f32(0.27886807),
            helper.const_f32(-0.18628806),
            helper.const_f32(0.09678418),
            helper.const_f32(0.37409196),
            helper.const_f32(1.00002368),
            helper.const_f32(-1.26551223)};
    auto p = polynomial(helper, t, coeff);

    auto r = helper.mul(t, helper.exp(helper.sub(p, helper.mul(x, x))));
    return helper.select(helper.ge(x, zero),
                         helper.sub(one, r),
                         helper.sub(r, one));
}

// numerical approximation of the inverse of normal distribution function
// original algorithm:
// https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtri.c
// case 1: 0 < x < exp(-2)
//     z = sqrt(-2 * log(x))
//     t = 1 / z
//     res = log(z) / z - z + t * P(t) / Q(t)
//     where coefficients of P and Q are different
//     for z < 8 and for z >= 8
//
// case2: exp(-2) <= x <= 1 - exp(-2)
//     w = x - 0.5
//     res = sqrt(2pi) * (w + w^3 * R(w^2) / S(w^2))
//
// case3: 1 - exp(-2) < x < 1
//     0 < 1 - x < exp(-2)
//     ndtri(x) = -ndtri(1 - x)
//     fallback to case 1
mlir::Value ndtri_approx(ValueBuilderHelper& helper, mlir::Value x) {
    // polynomial P
    auto P = [&](mlir::Value i, mlir::Value cond) {
        std::vector<mlir::Value> coeff0 = {
                helper.const_f32(4.05544892305962419923E0),
                helper.const_f32(3.15251094599893866154E1),
                helper.const_f32(5.71628192246421288162E1),
                helper.const_f32(4.40805073893200834700E1),
                helper.const_f32(1.46849561928858024014E1),
                helper.const_f32(2.18663306850790267539E0),
                helper.const_f32(-1.40256079171354495875E-1),
                helper.const_f32(-3.50424626827848203418E-2),
                helper.const_f32(-8.57456785154685413611E-4)};
        std::vector<mlir::Value> coeff1 = {
                helper.const_f32(3.23774891776946035970E0),
                helper.const_f32(6.91522889068984211695E0),
                helper.const_f32(3.93881025292474443415E0),
                helper.const_f32(1.33303460815807542389E0),
                helper.const_f32(2.01485389549179081538E-1),
                helper.const_f32(1.23716634817820021358E-2),
                helper.const_f32(3.01581553508235416007E-4),
                helper.const_f32(2.65806974686737550832E-6),
                helper.const_f32(6.23974539184983293730E-9)};
        return helper.select(cond,
                             polynomial(helper, i, coeff0),
                             polynomial(helper, i, coeff1));
    };

    // polynomial Q
    auto Q = [&](mlir::Value i, mlir::Value cond) {
        std::vector<mlir::Value> coeff0 = {
                helper.const_f32(1.f),
                helper.const_f32(1.57799883256466749731E1),
                helper.const_f32(4.53907635128879210584E1),
                helper.const_f32(4.13172038254672030440E1),
                helper.const_f32(1.50425385692907503408E1),
                helper.const_f32(2.50464946208309415979E0),
                helper.const_f32(-1.42182922854787788574E-1),
                helper.const_f32(-3.80806407691578277194E-2),
                helper.const_f32(-9.33259480895457427372E-4)};
        std::vector<mlir::Value> coeff1 = {
                helper.const_f32(1.f),
                helper.const_f32(6.02427039364742014255E0),
                helper.const_f32(3.67983563856160859403E0),
                helper.const_f32(1.37702099489081330271E0),
                helper.const_f32(2.16236993594496635890E-1),
                helper.const_f32(1.34204006088543189037E-2),
                helper.const_f32(3.28014464682127739104E-4),
                helper.const_f32(2.89247864745380683936E-6),
                helper.const_f32(6.79019408009981274425E-9)};
        return helper.select(cond,
                             polynomial(helper, i, coeff0),
                             polynomial(helper, i, coeff1));
    };

    // polynomial R
    auto R = [&](mlir::Value i) {
        std::vector<mlir::Value> coeff = {
                helper.const_f32(-5.99633501014107895267E1),
                helper.const_f32(9.80010754185999661536E1),
                helper.const_f32(-5.66762857469070293439E1),
                helper.const_f32(1.39312609387279679503E1),
                helper.const_f32(-1.23916583867381258016E0)};
        return polynomial(helper, i, coeff);
    };

    // polynomial S
    auto S = [&](mlir::Value i) {
        std::vector<mlir::Value> coeff = {
                helper.const_f32(1.f),
                helper.const_f32(1.95448858338141759834E0),
                helper.const_f32(4.67627912898881538453E0),
                helper.const_f32(8.63602421390890590575E1),
                helper.const_f32(-2.25462687854119370527E2),
                helper.const_f32(2.00260212380060660359E2),
                helper.const_f32(-8.20372256168333339912E1),
                helper.const_f32(1.59056225126211695515E1),
                helper.const_f32(-1.18331621121330003142E0)};
        return polynomial(helper, i, coeff);
    };

    // constants
    auto zero = helper.const_f32(0);
    auto one = helper.const_f32(1);
    auto half = helper.const_f32(0.5);
    auto eight = helper.const_f32(8);
    auto minus_2 = helper.const_f32(-2);
    auto exp_minus_2 = helper.const_f32(0.135335283236);  // exp(-2)
    auto sqrt_2pi = helper.const_f32(2.506628274631);     // sqrt(2pi)

    // conditions
    auto case1 = helper.lt(x, exp_minus_2);                   // x < exp(-2)
    auto case3 = helper.gt(x, helper.sub(one, exp_minus_2));  // x > 1 - exp(-2)
    auto case13 = helper.bit_or(case1, case3);

    // case1 or case3
    auto x13 = helper.select(case1, x, helper.sub(one, x));  // x or (1 - x)
    auto z = helper.sqrt(helper.mul(minus_2, helper.log(x13)));
    auto z_lt_8 = helper.lt(z, eight);
    auto t = helper.div(one, z);
    auto res1 = helper.add(helper.sub(helper.div(helper.log(z), z), z),
                           helper.div(helper.mul(t, P(t, z_lt_8)), Q(t, z_lt_8)));
    auto res13 = helper.select(case1, res1, helper.sub(zero, res1));

    // case2
    auto w = helper.sub(x, half);
    auto w2 = helper.mul(w, w);
    auto w3 = helper.mul(w, w2);
    auto res2 = helper.mul(
            sqrt_2pi, helper.add(w, helper.div(helper.mul(w3, R(w2)), S(w2))));

    return helper.select(case13, res13, res2);
}

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
