#ifndef COMPTON_KERNEL_APPROXIMATE_HPP
#define COMPTON_KERNEL_APPROXIMATE_HPP

/**
 * @file compton_kernel_approximate.hpp
 * @brief Fast order-five Sazonov thermal Compton kernel approximation.
 *
 * This implementation evaluates the explicit finite K_G5 coefficients with
 * two Padé approximants and the pole-suppressed blend proposed by Sazonov and
 * Sunyaev.  The six endpoint derivatives are closed-form expressions.  No jet
 * arithmetic, finite differences, logarithmic kernel, or dynamic allocation
 * is used.
 */

#include "compton_common/compton_common.hpp"

namespace compton {

class ComptonKernelApproximate {
  public:
    [[nodiscard]] ComptonResult
    sigma_E(double E, double E_prime, double xi, double T) const;

    [[nodiscard]] ComptonResult
    dsigma_E_dT(double E, double E_prime, double xi, double T) const;

  private:
    struct Evaluation {
        double value;
        double dvalue_dT;
        double estimated_rel_error;
    };

    [[nodiscard]] Evaluation
    evaluate(double E, double E_prime, double xi, double T) const;
};

} // namespace compton

#endif
