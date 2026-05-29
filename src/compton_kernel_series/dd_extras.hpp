#ifndef DD_EXTRAS_HPP
#define DD_EXTRAS_HPP
/**
 * @file dd_extras.hpp
 * @brief Adapter header for the WarrenWeckesser/doubledouble library.
 *
 * Provides a type alias (DD) and the two domain-specific functions not
 * present in the external library: dd_asinh and dd_ehat_cf.
 */

#include <cmath>
#include "../doubledouble.h"

namespace compton {

using DD = doubledouble::DoubleDouble;

inline double dd_to_double(const DD& x) { return x.upper + x.lower; }

/**
 * @brief Inverse hyperbolic sine in double-double precision.
 *
 * asinh(x) = log(x + sqrt(x^2 + 1))
 */
inline DD dd_asinh(const DD& x) {
    return (x + (x * x + 1.0).sqrt()).log();
}

/**
 * @brief Ehat_m(x) = exp(x) * E_m(x) via modified Lentz continued fraction.
 *
 * DLMF 8.9.2: Ehat_m(x) = 1/(x+m - m*1/(x+m+2 - (m+1)*2/(x+m+4 - ...)))
 * Evaluated via modified Lentz algorithm in double-double arithmetic.
 */
inline DD dd_ehat_cf(int m, const DD& x) {
    constexpr double TINY = 1e-300;
    constexpr double CF_TOL = 1e-31;
    constexpr int MAX_ITER = 200;

    DD b = x + static_cast<double>(m);
    if (std::abs(b.upper) < TINY) b.upper = TINY;

    DD f = b;
    DD C = b;
    DD D(0.0, 0.0);

    for (int j = 1; j <= MAX_ITER; ++j) {
        double aj = -static_cast<double>(m + j - 1) * static_cast<double>(j);
        DD bj = x + static_cast<double>(m + 2 * j);

        D = bj + D * aj;
        if (std::abs(D.upper) < TINY) D.upper = TINY;
        D = DD(1.0) / D;

        C = bj + DD(aj) / C;
        if (std::abs(C.upper) < TINY) C.upper = TINY;

        DD delta = C * D;
        f = f * delta;

        if (std::abs(delta.upper - 1.0) + std::abs(delta.lower) < CF_TOL)
            break;
    }

    return DD(1.0) / f;
}

} // namespace compton

#endif
