#pragma once

#include <cstring>
#include "src/common/utils.h"
#include "src/fallback/general_intrinsic/gi_float.h"

namespace megdnn {
namespace fallback {

template <typename ctype, size_t len>
struct Vector;

template <>
struct Vector<float, 4> {
    GI_FLOAT32_FIXLEN_t value;
    Vector() {}
    Vector(const float v) { value = GiFloat32Type2FixLenType(GiBroadcastFloat32(v)); }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const GI_FLOAT32_t& v) { value = GiFloat32Type2FixLenType(v); }
    static Vector load(const float* addr) {
        Vector v;
        v.value = GiFloat32Type2FixLenType(GiLoadFloat32(addr));
        return v;
    }
    static void save(float* addr, const Vector& v) {
        GiStoreFloat32(addr, GiFixLenType2GiFloat32Type(v.value));
    }
    void save(float* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value = GiFloat32Type2FixLenType(GiAddFloat32(
                GiFixLenType2GiFloat32Type(value),
                GiFixLenType2GiFloat32Type(lr.value)));
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value = GiFloat32Type2FixLenType(GiAddFloat32(
                GiFixLenType2GiFloat32Type(value),
                GiFixLenType2GiFloat32Type(lr.value)));
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value = GiFloat32Type2FixLenType(GiSubtractFloat32(
                GiFixLenType2GiFloat32Type(value),
                GiFixLenType2GiFloat32Type(lr.value)));
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value = GiFloat32Type2FixLenType(GiSubtractFloat32(
                GiFixLenType2GiFloat32Type(value),
                GiFixLenType2GiFloat32Type(lr.value)));
        return *this;
    }
    Vector operator*(float lr) {
        Vector dst;
        dst.value = GiFloat32Type2FixLenType(
                GiMultiplyScalerFloat32(GiFixLenType2GiFloat32Type(value), lr));
        return dst;
    }
    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value = GiFloat32Type2FixLenType(GiMultiplyFloat32(
                GiFixLenType2GiFloat32Type(value),
                GiFixLenType2GiFloat32Type(lr.value)));
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value = GiFloat32Type2FixLenType(GiMultiplyFloat32(
                GiFixLenType2GiFloat32Type(value),
                GiFixLenType2GiFloat32Type(lr.value)));
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value = -value;
        return dst;
    }
};

template <>
struct Vector<float, 8> {
    GI_FLOAT32_FIXLEN_V2_t value;
    Vector() {}
    Vector(const float v) {
        value.val[0] = GiFloat32Type2FixLenType(GiBroadcastFloat32(v));
        value.val[1] = GiFloat32Type2FixLenType(GiBroadcastFloat32(v));
    }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const GI_FLOAT32_V2_t& v) { value = GiFloat32Type2FixLenV2Type(v); }
    static Vector load(const float* addr) {
        Vector v;
        v.value = GiFloat32Type2FixLenV2Type(GiLoadFloat32V2(addr));
        return v;
    }
    static void save(float* addr, const Vector& v) {
        GiStoreFloat32V2(addr, GiFixLenType2GiFloat32V2Type(v.value));
    }

    void save(float* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = GiFloat32Type2FixLenType(GiAddFloat32(
                GiFixLenType2GiFloat32Type(value.val[0]),
                GiFixLenType2GiFloat32Type(lr.value.val[0])));
        dst.value.val[1] = GiFloat32Type2FixLenType(GiAddFloat32(
                GiFixLenType2GiFloat32Type(value.val[1]),
                GiFixLenType2GiFloat32Type(lr.value.val[1])));
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value.val[0] = GiFloat32Type2FixLenType(GiAddFloat32(
                GiFixLenType2GiFloat32Type(value.val[0]),
                GiFixLenType2GiFloat32Type(lr.value.val[0])));
        value.val[1] = GiFloat32Type2FixLenType(GiAddFloat32(
                GiFixLenType2GiFloat32Type(value.val[1]),
                GiFixLenType2GiFloat32Type(lr.value.val[1])));
        return *this;
    }
    Vector& add(const Vector& lr) {
        value.val[0] = GiFloat32Type2FixLenType(GiAddFloat32(
                GiFixLenType2GiFloat32Type(value.val[0]),
                GiFixLenType2GiFloat32Type(lr.value.val[0])));
        value.val[1] = GiFloat32Type2FixLenType(GiAddFloat32(
                GiFixLenType2GiFloat32Type(value.val[1]),
                GiFixLenType2GiFloat32Type(lr.value.val[1])));
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = GiFloat32Type2FixLenType(GiSubtractFloat32(
                GiFixLenType2GiFloat32Type(value.val[0]),
                GiFixLenType2GiFloat32Type(lr.value.val[0])));
        dst.value.val[1] = GiFloat32Type2FixLenType(GiSubtractFloat32(
                GiFixLenType2GiFloat32Type(value.val[1]),
                GiFixLenType2GiFloat32Type(lr.value.val[1])));
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value.val[0] = GiFloat32Type2FixLenType(GiSubtractFloat32(
                GiFixLenType2GiFloat32Type(value.val[0]),
                GiFixLenType2GiFloat32Type(lr.value.val[0])));
        value.val[1] = GiFloat32Type2FixLenType(GiSubtractFloat32(
                GiFixLenType2GiFloat32Type(value.val[1]),
                GiFixLenType2GiFloat32Type(lr.value.val[1])));
        return *this;
    }
    Vector operator*(float lr) {
        Vector dst;
        dst.value.val[0] = GiFloat32Type2FixLenType(
                GiMultiplyScalerFloat32(GiFixLenType2GiFloat32Type(value.val[0]), lr));
        dst.value.val[1] = GiFloat32Type2FixLenType(
                GiMultiplyScalerFloat32(GiFixLenType2GiFloat32Type(value.val[1]), lr));
        return dst;
    }
    //! val + lr * n
    Vector& mla(const Vector& lr, float n) {
        value.val[0] = GiFloat32Type2FixLenType(GiMultiplyAddScalarFloat32(
                GiFixLenType2GiFloat32Type(value.val[0]),
                GiFixLenType2GiFloat32Type(lr.value.val[0]), n));
        value.val[1] = GiFloat32Type2FixLenType(GiMultiplyAddScalarFloat32(
                GiFixLenType2GiFloat32Type(value.val[1]),
                GiFixLenType2GiFloat32Type(lr.value.val[1]), n));
        return *this;
    }

    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = GiFloat32Type2FixLenType(GiMultiplyFloat32(
                GiFixLenType2GiFloat32Type(value.val[0]),
                GiFixLenType2GiFloat32Type(lr.value.val[0])));
        dst.value.val[1] = GiFloat32Type2FixLenType(GiMultiplyFloat32(
                GiFixLenType2GiFloat32Type(value.val[1]),
                GiFixLenType2GiFloat32Type(lr.value.val[1])));
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value.val[0] = GiFloat32Type2FixLenType(GiMultiplyFloat32(
                GiFixLenType2GiFloat32Type(value.val[0]),
                GiFixLenType2GiFloat32Type(lr.value.val[0])));
        value.val[1] = GiFloat32Type2FixLenType(GiMultiplyFloat32(
                GiFixLenType2GiFloat32Type(value.val[1]),
                GiFixLenType2GiFloat32Type(lr.value.val[1])));
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value.val[0] = -value.val[0];
        dst.value.val[1] = -value.val[1];
        return dst;
    }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
