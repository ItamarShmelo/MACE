#include "compton_multigroup/weight_function.hpp"
#include "planck_integral.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

namespace compton {

CappedPlanckWeightFunction::CappedPlanckWeightFunction(double const cap_x)
    : cap_x_(cap_x),
      w0_(cap_x * cap_x * cap_x / std::expm1(cap_x))
{
    if (!(cap_x > 0.0)) {
        throw std::invalid_argument("cap_x must be > 0");
    }
}

double CappedPlanckWeightFunction::weight(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_) {
        return x * x * x / std::expm1(x);
    }
    return w0_;
}

double CappedPlanckWeightFunction::compute_denominator(
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

double CappedPlanckWeightFunction::d_weight_dT(double const E, double const T) const
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
CappedPlanckWeightFunction::d_log_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x >= cap_x_) {
        return 0.0;
    }
    double const ratio = x / (-std::expm1(-x));
    return -(3.0 - ratio) / T;
}

double CappedPlanckWeightFunction::d_denominator_dT(
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

std::optional<double> CappedPlanckWeightFunction::peak_energy(double const T) const
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

CappedWienWeightFunction::CappedWienWeightFunction(double const cap_x)
    : cap_x_(cap_x),
      w0_(cap_x * cap_x * cap_x * std::exp(-cap_x))
{
    if (!(cap_x > 0.0)) {
        throw std::invalid_argument("cap_x must be > 0");
    }
}

double CappedWienWeightFunction::weight(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_) {
        return x * x * x * std::exp(-x);
    }
    return w0_;
}

double CappedWienWeightFunction::compute_denominator(
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

double CappedWienWeightFunction::d_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x >= cap_x_) {
        return 0.0;
    }
    double const w = x * x * x * std::exp(-x);
    return -w * (3.0 - x) / T;
}

double
CappedWienWeightFunction::d_log_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x >= cap_x_) {
        return 0.0;
    }
    return -(3.0 - x) / T;
}

double CappedWienWeightFunction::d_denominator_dT(
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

std::optional<double> CappedWienWeightFunction::peak_energy(double const T) const
{
    return units::k_boltz * T * 3.0;
}

// ---------------------------------------------------------------------------
// Incomplete gamma helpers I_n(d) = int_0^d u^n exp(-u) du
// ---------------------------------------------------------------------------

namespace {

// Switchover threshold: for d <= 0.1, use Taylor; otherwise closed form.
constexpr double TAYLOR_THRESHOLD = 0.1;

double I_0(double const d)
{
    if (d <= TAYLOR_THRESHOLD) {
        return d *
               (1.0 -
                d / 2.0 *
                    (1.0 -
                     d / 3.0 *
                         (1.0 -
                          d / 4.0 *
                              (1.0 -
                               d / 5.0 *
                                   (1.0 -
                                    d / 6.0 * (1.0 - d / 7.0))))));
    }
    return 1.0 - std::exp(-d);
}

double I_1(double const d)
{
    if (d <= TAYLOR_THRESHOLD) {
        double const d2 = d * d;
        return d2 *
               (1.0 / 2.0 +
                d * (-1.0 / 3.0 +
                     d * (1.0 / 8.0 +
                          d * (-1.0 / 30.0 +
                               d * (1.0 / 144.0 +
                                    d * (-1.0 / 840.0 + d / 5760.0))))));
    }
    return 1.0 - std::exp(-d) * (1.0 + d);
}

double I_2(double const d)
{
    if (d <= TAYLOR_THRESHOLD) {
        double const d3 = d * d * d;
        return d3 *
               (1.0 / 3.0 +
                d * (-1.0 / 4.0 +
                     d * (1.0 / 10.0 +
                          d * (-1.0 / 36.0 +
                               d * (1.0 / 168.0 +
                                    d * (-1.0 / 960.0 + d / 6480.0))))));
    }
    return 2.0 - std::exp(-d) * (d * d + 2.0 * d + 2.0);
}

double I_3(double const d)
{
    if (d <= TAYLOR_THRESHOLD) {
        double const d4 = d * d * d * d;
        return d4 *
               (1.0 / 4.0 +
                d * (-1.0 / 5.0 +
                     d * (1.0 / 12.0 +
                          d * (-1.0 / 42.0 +
                               d * (1.0 / 192.0 +
                                    d * (-1.0 / 1080.0 + d / 7200.0))))));
    }
    return 6.0 - std::exp(-d) * (d * d * d + 3.0 * d * d + 6.0 * d + 6.0);
}

void validate_group_boundaries(std::vector<double> const& boundaries)
{
    if (boundaries.size() < 2) {
        throw std::invalid_argument(
            "group_boundaries must have at least 2 elements");
    }
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        if (!std::isfinite(boundaries[i]) || boundaries[i] < 0.0) {
            throw std::invalid_argument(
                "group_boundaries must be finite and nonnegative");
        }
        if (i > 0 && !(boundaries[i] > boundaries[i - 1])) {
            throw std::invalid_argument(
                "group_boundaries must be strictly increasing");
        }
    }
}

// Binary-search for the lower boundary of the group containing E.
// Uses std::upper_bound giving half-open groups [E_g, E_{g+1}).
// The final boundary is clamped so that E >= E_{G} belongs to the
// last group.
double find_x_lo_impl(
    std::vector<double> const& boundaries,
    double const E,
    double const T)
{
    auto const it =
        std::upper_bound(boundaries.begin(), boundaries.end(), E);
    auto const raw =
        static_cast<int>(std::distance(boundaries.begin(), it)) - 1;
    auto const g = std::clamp(
        raw, 0, static_cast<int>(boundaries.size()) - 2);
    return boundaries[static_cast<std::size_t>(g)] / (units::k_boltz * T);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// WienWeightFunction (shifted)
// ---------------------------------------------------------------------------

WienWeightFunction::WienWeightFunction(std::vector<double> group_boundaries)
    : boundaries_(std::move(group_boundaries))
{
    validate_group_boundaries(boundaries_);
}

double WienWeightFunction::find_x_lo(double const E, double const T) const
{
    return find_x_lo_impl(boundaries_, E, T);
}

double WienWeightFunction::weight(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    double const x_lo = find_x_lo(E, T);
    return x * x * x * std::exp(-(x - x_lo));
}

// compute_denominator assumes E_left and E_right are consecutive group
// boundaries so that x_lo = E_left/(kT) is the correct exponent shift.
double WienWeightFunction::compute_denominator(
    double const E_left,
    double const E_right,
    double const T) const
{
    double const kT = units::k_boltz * T;
    double const x_lo = E_left / kT;
    double const x_hi = E_right / kT;
    double const delta = x_hi - x_lo;

    double const result = x_lo * x_lo * x_lo * I_0(delta) +
                          3.0 * x_lo * x_lo * I_1(delta) +
                          3.0 * x_lo * I_2(delta) + I_3(delta);
    return kT * result;
}

double WienWeightFunction::d_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    double const x_lo = find_x_lo(E, T);
    double const w = x * x * x * std::exp(-(x - x_lo));
    return w * (x - x_lo - 3.0) / T;
}

double
WienWeightFunction::d_log_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    double const x_lo = find_x_lo(E, T);
    return (x - x_lo - 3.0) / T;
}

double WienWeightFunction::d_denominator_dT(
    double const E_left,
    double const E_right,
    double const T) const
{
    double const kT = units::k_boltz * T;
    double const x_lo = E_left / kT;
    double const x_hi = E_right / kT;

    double const D = compute_denominator(E_left, E_right, T);
    double const w_lo = x_lo * x_lo * x_lo;
    double const w_hi =
        x_hi * x_hi * x_hi * std::exp(-(x_hi - x_lo));

    return (D * (1.0 - x_lo) + E_left * w_lo - E_right * w_hi) / T;
}

std::optional<double> WienWeightFunction::peak_energy(double const T) const
{
    return units::k_boltz * T * 3.0;
}

// ---------------------------------------------------------------------------
// PlanckWeightFunction (shifted)
// ---------------------------------------------------------------------------

PlanckWeightFunction::PlanckWeightFunction(
    double const cap_x,
    std::vector<double> group_boundaries)
    : cap_x_(cap_x),
      boundaries_(std::move(group_boundaries))
{
    if (!std::isfinite(cap_x) || !(cap_x > 0.0)) {
        throw std::invalid_argument("cap_x must be finite and > 0");
    }
    validate_group_boundaries(boundaries_);
}

double PlanckWeightFunction::find_x_lo(double const E, double const T) const
{
    return find_x_lo_impl(boundaries_, E, T);
}

double PlanckWeightFunction::weight(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_) {
        return x * x * x / std::expm1(x);
    }
    double const x_lo = find_x_lo(E, T);
    if (x_lo >= cap_x_) {
        return x * x * x * std::exp(-(x - x_lo));
    }
    // Straddling group: x >= cap_x but x_lo < cap_x.
    // Use unshifted Wien (safe since x is near cap_x in this group).
    return x * x * x * std::exp(-x);
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
        double const delta = x_hi - x_lo;
        double const result = x_lo * x_lo * x_lo * I_0(delta) +
                              3.0 * x_lo * x_lo * I_1(delta) +
                              3.0 * x_lo * I_2(delta) + I_3(delta);
        return kT * result;
    }

    // Straddling: x_lo < cap_x_ < x_hi.
    // Planck from x_lo to cap_x, unshifted Wien from cap_x to x_hi.
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

    double const planck_part =
        pi4_over_15 * planck_integral::planck_integral(x_lo, cap_x_);
    double const wien_part = wien_antideriv(x_hi) - wien_antideriv(cap_x_);
    return kT * (planck_part + wien_part);
}

double PlanckWeightFunction::d_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_) {
        double const w = x * x * x / std::expm1(x);
        double const ratio = x / (-std::expm1(-x));
        return -w * (3.0 - ratio) / T;
    }
    double const x_lo = find_x_lo(E, T);
    if (x_lo >= cap_x_) {
        double const w = x * x * x * std::exp(-(x - x_lo));
        return w * (x - x_lo - 3.0) / T;
    }
    // Straddling: unshifted Wien
    double const w = x * x * x * std::exp(-x);
    return -w * (3.0 - x) / T;
}

double
PlanckWeightFunction::d_log_weight_dT(double const E, double const T) const
{
    double const x = E / (units::k_boltz * T);
    if (x < cap_x_) {
        double const ratio = x / (-std::expm1(-x));
        return -(3.0 - ratio) / T;
    }
    double const x_lo = find_x_lo(E, T);
    if (x_lo >= cap_x_) {
        return (x - x_lo - 3.0) / T;
    }
    return -(3.0 - x) / T;
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

    if (x_hi <= cap_x_) {
        double const D =
            kT * pi4_over_15 * planck_integral::planck_integral(x_lo, x_hi);
        double const w_lo = x_lo * x_lo * x_lo / std::expm1(x_lo);
        double const w_hi = x_hi * x_hi * x_hi / std::expm1(x_hi);
        return (D + E_left * w_lo - E_right * w_hi) / T;
    }

    if (x_lo >= cap_x_) {
        double const D = compute_denominator(E_left, E_right, T);
        double const w_lo = x_lo * x_lo * x_lo;
        double const w_hi =
            x_hi * x_hi * x_hi * std::exp(-(x_hi - x_lo));
        return (D * (1.0 - x_lo) + E_left * w_lo - E_right * w_hi) / T;
    }

    // Straddling: x_lo < cap_x_ < x_hi
    double const D = compute_denominator(E_left, E_right, T);
    double const w_planck_lo = x_lo * x_lo * x_lo / std::expm1(x_lo);
    double const w_wien_hi = x_hi * x_hi * x_hi * std::exp(-x_hi);
    return (D + E_left * w_planck_lo - E_right * w_wien_hi) / T;
}

std::optional<double> PlanckWeightFunction::peak_energy(double const T) const
{
    static constexpr double PLANCK_PEAK_X = 2.821439372122078893;
    return units::k_boltz * T * PLANCK_PEAK_X;
}

} // namespace compton
