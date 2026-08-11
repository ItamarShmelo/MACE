#ifndef COMPTON_KERNEL_APPROXIMATE_HPP
#define COMPTON_KERNEL_APPROXIMATE_HPP

#include "compton_common/compton_common.hpp"

namespace compton {

/**
 * @brief Fifth-order global Compton kernel approximation (KG5).
 *
 * Evaluates the angle-dependent thermal Compton redistribution kernel
 * using the explicit Section-9 endpoint coefficients, [3/2] and [2/3]
 * Padé continuations, and a pole-suppressed rational blend.
 *
 * The approximation is quadrature-free and constant-time per evaluation.
 * It reports zero self-error; accuracy is determined externally by
 * comparison scripts against ComptonKernelSolver.
 *
 * Error handling follows the existing solver convention: failures are
 * caught internally and reported as ComptonResult{0, 1, 0, 0}.
 */
class ComptonKernelApproximate {
  public:
    ComptonKernelApproximate() = default;

    ComptonResult sigma_E(double E, double E_prime, double xi, double T) const;

  private:
    double
    evaluate_sigma_E(double gamma, double gamma_prime, double xi, double tau)
        const;
};

} // namespace compton

#endif
