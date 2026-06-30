#ifndef WEIGHT_FUNCTION_HPP
#define WEIGHT_FUNCTION_HPP

#include <optional>

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

    virtual double weight(double E, double T) const = 0;

    virtual double
    compute_denominator(double E_left, double E_right, double T) const = 0;

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
class PlanckWeightFunction : public WeightFunction {
  public:
    explicit PlanckWeightFunction(double cap_x);

    double weight(double E, double T) const override;

    double
    compute_denominator(double E_left, double E_right, double T) const override;

    std::optional<double> peak_energy(double T) const override;

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
class WienWeightFunction : public WeightFunction {
  public:
    explicit WienWeightFunction(double cap_x);

    double weight(double E, double T) const override;

    double
    compute_denominator(double E_left, double E_right, double T) const override;

    std::optional<double> peak_energy(double T) const override;

  private:
    double cap_x_;
    double w0_;
};

} // namespace compton

#endif
