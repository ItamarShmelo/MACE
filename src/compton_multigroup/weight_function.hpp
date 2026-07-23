#ifndef WEIGHT_FUNCTION_HPP
#define WEIGHT_FUNCTION_HPP

#include <optional>
#include <vector>

namespace compton {

/**
 * @brief Abstract base for energy-group weight functions.
 *
 * A weight function w(E, T) defines how incident photon energies are
 * weighted when collapsing the continuous Compton kernel onto a multigroup
 * structure.  Subclasses provide the point-wise weight group-integrated
 * denominator.
 */
class WeightFunction {
  public:
    virtual ~WeightFunction() = default;
    WeightFunction() = default;
    WeightFunction(WeightFunction const&) = delete;
    WeightFunction& operator=(WeightFunction const&) = delete;
    WeightFunction(WeightFunction&&) = delete;
    WeightFunction& operator=(WeightFunction&&) = delete;

    virtual double weight(double E, double T) const = 0;

    virtual double
    compute_denominator(double E_left, double E_right, double T) const = 0;

    /// Partial derivative dw/dT of the point-wise weight.
    virtual double d_weight_dT(double E, double T) const = 0;

    /// Logarithmic derivative d(ln w)/dT, computed directly from the analytic
    /// formula without dividing by w. 
    virtual double d_log_weight_dT(double E, double T) const = 0;

    /// Temperature derivative of the group-integrated denominator,
    /// computed analytically via the Leibniz rule.
    virtual double
    d_denominator_dT(double E_left, double E_right, double T) const = 0;

    /// Energy [erg] at which w(E,T) attains its maximum, or nullopt if the
    /// weight has no interior peak (e.g. uniform).  Used by the multigroup
    /// integrator to place a panel boundary at the weight peak.
    virtual std::optional<double> peak_energy(double T) const = 0;
};

/**
 * @brief Capped Planck weight w(E, T) and its group-integrated denominator.
 *
 * The weight function is:
 *
 *     w(E,T) = x^3/(e^x - 1)   for x = E/(kT) < cap_x
 *            = w0               for x >= cap_x
 *
 * where w0 = cap_x^3/(e^{cap_x} - 1) ensures continuity at the cap.
 *
 * The denominator integral int_{E_left}^{E_right} w(E,T) dE is computed
 * analytically using the Clark (1987) polylogarithm series for the
 * below-cap portion and a constant for the above-cap portion.
 */
class CappedPlanckWeightFunction : public WeightFunction {
  public:
    explicit CappedPlanckWeightFunction(double cap_x);

    double weight(double E, double T) const override;

    double
    compute_denominator(double E_left, double E_right, double T) const override;

    double d_weight_dT(double E, double T) const override;
    double d_log_weight_dT(double E, double T) const override;
    double d_denominator_dT(double E_left, double E_right, double T) const override;

    std::optional<double> peak_energy(double T) const override;

    double cap_x() const { return cap_x_; }

  private:
    double cap_x_;
    double w0_;
};

/**
 * @brief Uniform (flat) weight: w(E, T) = 1.
 *
 * The denominator is simply E_right - E_left.
 */
class UniformWeightFunction : public WeightFunction {
  public:
    double weight(double E, double T) const override;

    double
    compute_denominator(double E_left, double E_right, double T) const override;

    double d_weight_dT(double E, double T) const override;
    double d_log_weight_dT(double E, double T) const override;
    double d_denominator_dT(double E_left, double E_right, double T) const override;

    std::optional<double> peak_energy(double T) const override;
};

/**
 * @brief Capped Wien weight w(E, T) and its group-integrated denominator.
 *
 * The weight function is:
 *
 *     w(E,T) = x^3 * exp(-x)   for x = E/(kT) < cap_x
 *            = w0               for x >= cap_x
 *
 * where w0 = cap_x^3 * exp(-cap_x) ensures continuity at the cap.
 */
class CappedWienWeightFunction : public WeightFunction {
  public:
    explicit CappedWienWeightFunction(double cap_x);

    double weight(double E, double T) const override;

    double
    compute_denominator(double E_left, double E_right, double T) const override;

    double d_weight_dT(double E, double T) const override;
    double d_log_weight_dT(double E, double T) const override;
    double d_denominator_dT(double E_left, double E_right, double T) const override;

    std::optional<double> peak_energy(double T) const override;

    double cap_x() const { return cap_x_; }

  private:
    double cap_x_;
    double w0_;
};

/**
 * @brief Shifted Wien weight using per-group exponent shift.
 *
 * The weight function is:
 *
 *     w(E,T) = x^3 * exp(-(x - x_lo))
 *
 * where x = E/(kT) and x_lo = E_g/(kT) is the lower boundary of the
 * energy group containing E.  The shift factor exp(x_lo) prevents
 * underflow at large x while cancelling in the numerator/denominator
 * ratio.
 *
 * @note The group_boundaries must match the energy_group_boundaries
 * passed to ComptonMultigroupKernel or ComptonMonteCarloKernel. A mismatch
 * causes the per-group exponent shift in the numerator integration to
 * differ from the shift used in the denominator, producing incorrect
 * multigroup cross-sections.
 */
class WienWeightFunction : public WeightFunction {
  public:
    explicit WienWeightFunction(std::vector<double> group_boundaries);

    double weight(double E, double T) const override;

    double
    compute_denominator(double E_left, double E_right, double T) const override;

    double d_weight_dT(double E, double T) const override;
    double d_log_weight_dT(double E, double T) const override;
    double
    d_denominator_dT(double E_left, double E_right, double T) const override;

    /// Stationary point of the underlying spectral shape (x=3 for Wien),
    /// used as a quadrature panel-splitting hint.  Not the global maximum
    /// of the piecewise shifted weight.
    std::optional<double> peak_energy(double T) const override;

  private:
    std::vector<double> boundaries_;
    double find_x_lo(double E, double T) const;
};

/**
 * @brief Shifted Planck weight using per-group exponent shift above cap_x.
 *
 * Below cap_x the true Planck weight x^3/(e^x - 1) is used.
 * Above cap_x the shifted Wien approximation x^3 * exp(-(x - x_lo))
 * is used, where x_lo is the lower boundary of the energy group
 * containing E in dimensionless units.  For groups that straddle cap_x,
 * the unshifted Wien x^3 * exp(-x) is used above cap_x (safe since
 * exp(-x) is still representable near cap_x).
 *
 * @note The group_boundaries must match the energy_group_boundaries
 * passed to ComptonMultigroupKernel or ComptonMonteCarloKernel. A mismatch
 * causes the per-group exponent shift in the numerator integration to
 * differ from the shift used in the denominator, producing incorrect
 * multigroup cross-sections.
 */
class PlanckWeightFunction : public WeightFunction {
  public:
    PlanckWeightFunction(double cap_x, std::vector<double> group_boundaries);

    double weight(double E, double T) const override;

    double
    compute_denominator(double E_left, double E_right, double T) const override;

    double d_weight_dT(double E, double T) const override;
    double d_log_weight_dT(double E, double T) const override;
    double
    d_denominator_dT(double E_left, double E_right, double T) const override;

    /// Stationary point of the underlying spectral shape (x=2.8214 for
    /// Planck), used as a quadrature panel-splitting hint.  Not the global
    /// maximum of the piecewise shifted weight.
    std::optional<double> peak_energy(double T) const override;

    double cap_x() const { return cap_x_; }

  private:
    double cap_x_;
    std::vector<double> boundaries_;
    double find_x_lo(double E, double T) const;
};

} // namespace compton

#endif
