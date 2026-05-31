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
    /**
     * @param method   Series expansion strategy (PowerSeries, Asymptotic, or
     *                 Auto which selects based on tau*alpha < 0.05).
     * @param eps_rel  Relative convergence tolerance for both term-decay
     *                 and difference-stability checks.
     * @param n_min    Minimum number of terms before convergence is tested.
     * @param n_max    Maximum number of terms; if reached without convergence
     *                 the result is returned with converged=false.
     */
    ComptonKernelSeries(
        SeriesMethod method = SeriesMethod::Auto,
        double eps_rel = 1e-12,
        int n_min = 4,
        int n_max = 200
    );

    SeriesResult sigma_E(
        double E,
        double E_prime,
        double xi,
        double tau,
        double Ne
    ) const;

private:
    template<typename T>
    SeriesResult power_series(
        double gamma,
        double gamma_p,
        double xi,
        double tau,
        double E,
        double Ne
    ) const;

    SeriesResult asymptotic_series(
        double gamma,
        double gamma_p,
        double xi,
        double tau,
        double E,
        double Ne
    ) const;

    SeriesMethod method_;
    double eps_rel_;
    int n_min_, n_max_;

};

} // namespace compton

#endif
