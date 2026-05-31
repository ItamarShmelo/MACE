#ifndef COMPTON_KERNEL_SERIES_HPP
#define COMPTON_KERNEL_SERIES_HPP
/**
 * @file compton_kernel_series.hpp
 * @brief Compton scattering kernel via Section 4 series expansions.
 *
 * Two series methods:
 *   - PowerSeries:  Poisson-weighted sum of scaled exponential integrals Ehat_m
 *   - Asymptotic:   Low-temperature expansion using Legendre polynomials
 *   - Auto:         Selects method based on tau*alpha threshold (0.05)
 *
 * The series module does NOT depend on the quadrature module.  If the chosen
 * series fails to converge, it returns converged=false and the caller decides
 * whether to fall back to quadrature.
 */

#include "compton_common/compton_common.hpp"

namespace compton {

enum class SeriesMethod {
    PowerSeries,
    Asymptotic,
    Auto
};

struct SeriesResult {
    double value;
    double estimated_abs_error;
    double estimated_rel_error;
    int terms_used;
    SeriesMethod method_used;
    bool converged;
};

class ComptonKernelSeries {
public:
    ComptonKernelSeries(
        SeriesMethod method = SeriesMethod::Auto,
        double eps_rel = 1e-12,
        int n_min = 4,
        int n_max = 200
    );

    SeriesResult sigma_E(double E, double E_prime, double xi,
                         double tau, double Ne) const;

private:
    SeriesResult power_series(const KershawParams<double>& p,
                              double gamma, double gamma_p, double xi,
                              double tau, double sigma0) const;
    SeriesResult asymptotic_series(const KershawParams<double>& p,
                                   double gamma, double gamma_p,
                                   double tau, double sigma0) const;
    SeriesMethod method_;
    double eps_rel_;
    int n_min_, n_max_;

};

} // namespace compton

#endif
