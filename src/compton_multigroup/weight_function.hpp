#ifndef WEIGHT_FUNCTION_HPP
#define WEIGHT_FUNCTION_HPP

namespace compton {

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
class PlanckWeightFunction {
public:
    explicit PlanckWeightFunction(double cap_x);

    double weight(double E, double T) const;

    double compute_denominator(double E_left, double E_right, double T) const;

private:
    double cap_x_;
    double w0_;
};

} // namespace compton

#endif
