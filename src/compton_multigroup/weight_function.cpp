#include "compton_multigroup/weight_function.hpp"
#include "planck_integral.hpp"
#include "units/units.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace compton {

PlanckWeightFunction::PlanckWeightFunction(double const cap_x)
    : cap_x_(cap_x)
    , w0_(cap_x * cap_x * cap_x / std::expm1(cap_x))
{
    if (!(cap_x > 0.0))
        throw std::invalid_argument("cap_x must be > 0");
}

double PlanckWeightFunction::weight(double const E, double const T) const {
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_)
        return x * x * x / std::expm1(x);
    return w0_;
}

double PlanckWeightFunction::compute_denominator(
    double const E_left, double const E_right, double const T) const
{
    double const kT = units::k_boltz * T;
    double const x_lo = E_left / kT;
    double const x_hi = E_right / kT;

    static double constexpr pi4_over_15 =
        std::numbers::pi * std::numbers::pi *
        std::numbers::pi * std::numbers::pi / 15.0;

    if (x_hi <= cap_x_)
        return kT * pi4_over_15 * planck_integral::planck_integral(x_lo, x_hi);

    if (x_lo >= cap_x_)
        return kT * w0_ * (x_hi - x_lo);

    double const planck_part = pi4_over_15 *
        planck_integral::planck_integral(x_lo, cap_x_);
    double const const_part = w0_ * (x_hi - cap_x_);
    return kT * (planck_part + const_part);
}

} // namespace compton
