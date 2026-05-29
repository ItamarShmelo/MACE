#ifndef DOUBLE_DOUBLE_HPP
#define DOUBLE_DOUBLE_HPP
/**
 * @file double_double.hpp
 * @brief Header-only double-double (dd) arithmetic library.
 *
 * Provides ~32-digit precision using pairs of IEEE 754 doubles.
 * Layers: core ops, exp, log, sqrt, asinh, ehat_cf (Lentz CF for
 * the scaled exponential integral Ehat_m).
 */

#include <cmath>
#include <limits>
#include <algorithm>

namespace compton {

// ═══════════════════════════════════════════════════════════════════════════
// Layer 1: Core dd arithmetic
// ═══════════════════════════════════════════════════════════════════════════

struct dd { double hi, lo; };

inline dd dd_from_double(double x) { return {x, 0.0}; }
inline double dd_to_double(dd x)   { return x.hi + x.lo; }

inline dd two_sum(double a, double b) {
    double s = a + b;
    double v = s - a;
    double e = (a - (s - v)) + (b - v);
    return {s, e};
}

inline dd two_prod(double a, double b) {
    double p = a * b;
    double e = std::fma(a, b, -p);
    return {p, e};
}

inline dd dd_add(dd a, double b) {
    dd s = two_sum(a.hi, b);
    s.lo += a.lo;
    return two_sum(s.hi, s.lo);
}

inline dd dd_add(dd a, dd b) {
    dd s = two_sum(a.hi, b.hi);
    dd t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return two_sum(s.hi, s.lo);
}

inline dd dd_sub(dd a, dd b) {
    return dd_add(a, {-b.hi, -b.lo});
}

inline dd dd_mul(dd a, dd b) {
    dd p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return two_sum(p.hi, p.lo);
}

inline dd dd_mul_scalar(dd a, double b) {
    dd p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    return two_sum(p.hi, p.lo);
}

inline dd dd_div(dd a, dd b) {
    double q1 = a.hi / b.hi;
    dd r = dd_sub(a, dd_mul_scalar(b, q1));
    double q2 = r.hi / b.hi;
    r = dd_sub(r, dd_mul_scalar(b, q2));
    double q3 = r.hi / b.hi;
    dd q = two_sum(q1, q2);
    return dd_add(q, q3);
}

inline dd dd_div_scalar(dd a, double b) {
    return dd_div(a, dd_from_double(b));
}

inline dd dd_abs(dd a) {
    return (a.hi < 0.0) ? dd{-a.hi, -a.lo} : a;
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 2: dd_exp
// ═══════════════════════════════════════════════════════════════════════════

namespace detail {
    // ln(2) split into hi + lo for exact range reduction
    static constexpr double LN2_HI = 6.931471805599452862e-01;
    static constexpr double LN2_LO = 2.319046813023978449e-17;
    static constexpr dd DD_LN2 = {LN2_HI, LN2_LO};
} // namespace detail

inline dd dd_exp(dd a) {
    if (a.hi > 700.0) return {std::numeric_limits<double>::infinity(), 0.0};
    if (a.hi < -700.0) return {0.0, 0.0};

    double k = std::round(a.hi / detail::LN2_HI);
    dd r = dd_sub(a, dd_mul_scalar(detail::DD_LN2, k));

    // Taylor series: exp(r) = 1 + r + r^2/2! + ... (12 terms)
    dd sum = dd_from_double(1.0);
    dd term = dd_from_double(1.0);
    for (int i = 1; i <= 12; ++i) {
        term = dd_div_scalar(dd_mul(term, r), static_cast<double>(i));
        sum = dd_add(sum, term);
        if (std::abs(term.hi) < 1e-32 * std::abs(sum.hi))
            break;
    }

    // Scale by 2^k via std::ldexp (exact)
    int ki = static_cast<int>(k);
    sum.hi = std::ldexp(sum.hi, ki);
    sum.lo = std::ldexp(sum.lo, ki);
    return sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 3: dd_log
// ═══════════════════════════════════════════════════════════════════════════

inline dd dd_log(dd a) {
    double y0 = std::log(a.hi);
    // One Newton step: y1 = y0 + (a * exp(-y0) - 1)
    dd ey = dd_exp({-y0, 0.0});
    dd aey = dd_mul(a, ey);
    dd corr = dd_sub(aey, dd_from_double(1.0));
    return dd_add(dd_from_double(y0), corr);
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 4: dd_sqrt
// ═══════════════════════════════════════════════════════════════════════════

inline dd dd_sqrt(dd a) {
    if (a.hi <= 0.0 && a.lo == 0.0) return {0.0, 0.0};
    double y0 = std::sqrt(a.hi);
    // Newton: y1 = (y0 + a/y0) / 2
    dd q = dd_div(a, dd_from_double(y0));
    return dd_mul_scalar(dd_add(dd_from_double(y0), q), 0.5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 5: dd_asinh
// ═══════════════════════════════════════════════════════════════════════════

inline dd dd_asinh(dd x) {
    // asinh(x) = log(x + sqrt(x^2 + 1))
    dd x2 = dd_mul(x, x);
    dd arg = dd_sqrt(dd_add(x2, 1.0));
    return dd_log(dd_add(x, arg));
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 6: dd_ehat_cf -- Ehat_m(x) via modified Lentz continued fraction
// ═══════════════════════════════════════════════════════════════════════════
//
// DLMF 8.9.2: Ehat_m(x) = 1/(x+m - m·1/(x+m+2 - (m+1)·2/(x+m+4 - ...)))
// Evaluated via modified Lentz algorithm in dd arithmetic.

inline dd dd_ehat_cf(int m, dd x) {
    constexpr double TINY = 1e-300;
    constexpr double CF_TOL = 1e-31;
    constexpr int MAX_ITER = 200;

    // b_0 = x + m
    dd b = dd_add(x, static_cast<double>(m));
    if (std::abs(b.hi) < TINY) b.hi = TINY;

    dd f = b;
    dd C = b;
    dd D = {0.0, 0.0};

    for (int j = 1; j <= MAX_ITER; ++j) {
        // a_j = -(m + j - 1) * j
        double aj = -static_cast<double>(m + j - 1) * static_cast<double>(j);
        // b_j = x + m + 2*j
        dd bj = dd_add(x, static_cast<double>(m + 2 * j));

        D = dd_add(bj, dd_mul_scalar(D, aj));
        if (std::abs(D.hi) < TINY) D.hi = TINY;
        D = dd_div(dd_from_double(1.0), D);

        C = dd_add(bj, dd_div(dd_from_double(aj), C));
        if (std::abs(C.hi) < TINY) C.hi = TINY;

        dd delta = dd_mul(C, D);
        f = dd_mul(f, delta);

        if (std::abs(delta.hi - 1.0) + std::abs(delta.lo) < CF_TOL)
            break;
    }

    return dd_div(dd_from_double(1.0), f);
}

} // namespace compton

#endif
