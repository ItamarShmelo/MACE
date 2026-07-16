#include "compton_multigroup/weight_function.hpp"
#include "planck_integral.hpp"
#include "utilities/units.hpp"

#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>

namespace compton {

PlanckWeightFunction::PlanckWeightFunction(double const cap_x)
    : cap_x_(cap_x),
      w0_(cap_x * cap_x * cap_x / std::expm1(cap_x))
{
    if (!(cap_x > 0.0)) {
        throw std::invalid_argument("cap_x must be > 0");
    }
}

double PlanckWeightFunction::weight(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_) {
        return x * x * x / std::expm1(x);
    }
    return w0_;
}

double PlanckWeightFunction::compute_denominator(
    double const E_left,
    double const E_right,
    double const T) const
{
    double const kT = units::k_boltz * T;
    double const x_lo = E_left / kT;
    double const x_hi = E_right / kT;

    static constexpr double pi4_over_15 = std::numbers::pi * std::numbers::pi *
                                          std::numbers::pi * std::numbers::pi /
                                          15.0;

    if (x_hi <= cap_x_) {
        return kT * pi4_over_15 * planck_integral::planck_integral(x_lo, x_hi);
    }

    if (x_lo >= cap_x_) {
        return kT * w0_ * (x_hi - x_lo);
    }

    double const planck_part =
        pi4_over_15 * planck_integral::planck_integral(x_lo, cap_x_);
    double const const_part = w0_ * (x_hi - cap_x_);
    return kT * (planck_part + const_part);
}

double PlanckWeightFunction::d_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x >= cap_x_) {
        return 0.0;
    }
    double const w = x * x * x / std::expm1(x);
    double const ratio = x / (-std::expm1(-x));
    return -w * (3.0 - ratio) / T;
}

double
PlanckWeightFunction::d_log_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x >= cap_x_) {
        return 0.0;
    }
    double const ratio = x / (-std::expm1(-x));
    return -(3.0 - ratio) / T;
}

double PlanckWeightFunction::d_denominator_dT(
    double const E_left,
    double const E_right,
    double const T) const
{
    double const kT = units::k_boltz * T;
    double const x_lo = E_left / kT;
    double const x_hi = E_right / kT;

    static constexpr double pi4_over_15 = std::numbers::pi * std::numbers::pi *
                                          std::numbers::pi * std::numbers::pi /
                                          15.0;

    if (x_lo >= cap_x_) {
        return 0.0;
    }

    if (x_hi <= cap_x_) {
        double const D =
            kT * pi4_over_15 * planck_integral::planck_integral(x_lo, x_hi);
        double const w_lo = x_lo * x_lo * x_lo / std::expm1(x_lo);
        double const w_hi = x_hi * x_hi * x_hi / std::expm1(x_hi);
        return (D + E_left * w_lo - E_right * w_hi) / T;
    }

    // Straddling: x_lo < cap_x_ < x_hi
    double const D_uncapped =
        kT * pi4_over_15 * planck_integral::planck_integral(x_lo, cap_x_);
    double const w_lo = x_lo * x_lo * x_lo / std::expm1(x_lo);
    double const E_cap = cap_x_ * kT;
    return (D_uncapped + E_left * w_lo - E_cap * w0_) / T;
}

std::optional<double> PlanckWeightFunction::peak_energy(double const T) const
{
    static constexpr double PLANCK_PEAK_X = 2.821439372122078893;
    return units::k_boltz * T * PLANCK_PEAK_X;
}

double
UniformWeightFunction::weight(double const /*E*/, double const /*T*/) const
{
    return 1.0;
}

double UniformWeightFunction::compute_denominator(
    double const E_left,
    double const E_right,
    double const /*T*/) const
{
    return E_right - E_left;
}

double UniformWeightFunction::d_weight_dT(
    double const /*E*/,
    double const /*T*/) const
{
    return 0.0;
}

double UniformWeightFunction::d_log_weight_dT(
    double const /*E*/,
    double const /*T*/) const
{
    return 0.0;
}

double UniformWeightFunction::d_denominator_dT(
    double const /*E_left*/,
    double const /*E_right*/,
    double const /*T*/) const
{
    return 0.0;
}

std::optional<double>
UniformWeightFunction::peak_energy(double const /*T*/) const
{
    return std::nullopt;
}

WienWeightFunction::WienWeightFunction(double const cap_x)
    : cap_x_(cap_x),
      w0_(cap_x * cap_x * cap_x * std::exp(-cap_x))
{
    if (!(cap_x > 0.0)) {
        throw std::invalid_argument("cap_x must be > 0");
    }
}

double WienWeightFunction::weight(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_) {
        return x * x * x * std::exp(-x);
    }
    return w0_;
}

double WienWeightFunction::compute_denominator(
    double const E_left,
    double const E_right,
    double const T) const
{
    double const kT = units::k_boltz * T;
    double const x_lo = E_left / kT;
    double const x_hi = E_right / kT;

    // G(x) = int_0^x t^3 e^{-t} dt.  Taylor branch for x <= 0.1 avoids
    // catastrophic cancellation in 6 - e^{-x}(x^3+3x^2+6x+6).
    // 7-term Horner gives relative error < 1e-9 at x = 0.1.
    auto const wien_antideriv = [](double const x) {
        if (x <= 0.1) {
            double const x4 = x * x * x * x;
            return x4 *
                   (1.0 / 4.0 +
                    x * (-1.0 / 5.0 +
                         x * (1.0 / 12.0 +
                              x * (-1.0 / 42.0 +
                                   x * (1.0 / 192.0 +
                                        x * (-1.0 / 1080.0 + x / 7200.0))))));
        }
        return 6.0 - std::exp(-x) * (x * x * x + 3.0 * x * x + 6.0 * x + 6.0);
    };

    if (x_hi <= cap_x_) {
        return kT * (wien_antideriv(x_hi) - wien_antideriv(x_lo));
    }

    if (x_lo >= cap_x_) {
        return kT * w0_ * (x_hi - x_lo);
    }

    double const wien_part = wien_antideriv(cap_x_) - wien_antideriv(x_lo);
    double const const_part = w0_ * (x_hi - cap_x_);
    return kT * (wien_part + const_part);
}

double WienWeightFunction::d_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x >= cap_x_) {
        return 0.0;
    }
    double const w = x * x * x * std::exp(-x);
    return -w * (3.0 - x) / T;
}

double
WienWeightFunction::d_log_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x >= cap_x_) {
        return 0.0;
    }
    return -(3.0 - x) / T;
}

double WienWeightFunction::d_denominator_dT(
    double const E_left,
    double const E_right,
    double const T) const
{
    double const kT = units::k_boltz * T;
    double const x_lo = E_left / kT;
    double const x_hi = E_right / kT;

    auto const wien_antideriv = [](double const x) {
        if (x <= 0.1) {
            double const x4 = x * x * x * x;
            return x4 *
                   (1.0 / 4.0 +
                    x * (-1.0 / 5.0 +
                         x * (1.0 / 12.0 +
                              x * (-1.0 / 42.0 +
                                   x * (1.0 / 192.0 +
                                        x * (-1.0 / 1080.0 + x / 7200.0))))));
        }
        return 6.0 - std::exp(-x) * (x * x * x + 3.0 * x * x + 6.0 * x + 6.0);
    };

    if (x_lo >= cap_x_) {
        return 0.0;
    }

    if (x_hi <= cap_x_) {
        double const D = kT * (wien_antideriv(x_hi) - wien_antideriv(x_lo));
        double const w_lo = x_lo * x_lo * x_lo * std::exp(-x_lo);
        double const w_hi = x_hi * x_hi * x_hi * std::exp(-x_hi);
        return (D + E_left * w_lo - E_right * w_hi) / T;
    }

    // Straddling: x_lo < cap_x_ < x_hi
    double const D_uncapped =
        kT * (wien_antideriv(cap_x_) - wien_antideriv(x_lo));
    double const w_lo = x_lo * x_lo * x_lo * std::exp(-x_lo);
    double const E_cap = cap_x_ * kT;
    return (D_uncapped + E_left * w_lo - E_cap * w0_) / T;
}

std::optional<double> WienWeightFunction::peak_energy(double const T) const
{
    return units::k_boltz * T * 3.0;
}

} // namespace compton
