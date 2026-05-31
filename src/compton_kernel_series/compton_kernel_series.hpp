#ifndef COMPTON_KERNEL_SERIES_HPP
#define COMPTON_KERNEL_SERIES_HPP
/**
 * @file compton_kernel_series.hpp
 * @brief Compton scattering kernel via Section 4 series expansions.
 *
 * Series methods:
 *   - PowerSeries:              double-precision power series (~15 digits)
 *   - PowerSeriesHighPrecision: double-double power series   (~31 digits)
 *   - Asymptotic:               Low-temperature expansion using Legendre polynomials
 *   - Auto:                     Selects Asymptotic or PowerSeriesHighPrecision
 *                               based on tau*alpha threshold (0.05)
 *
 * The series module does NOT depend on the quadrature module.  If the chosen
 * series fails to converge, it returns converged=false and the caller decides
 * whether to fall back to quadrature.
 */


namespace compton {

enum class SeriesMethod {
    PowerSeries,
    PowerSeriesHighPrecision,
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
     * @param method   Series expansion strategy (PowerSeries,
     *                 PowerSeriesHighPrecision, Asymptotic, or Auto which
     *                 selects based on tau*alpha < 0.05).
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
        double const E,
        double const E_prime,
        double const xi,
        double const tau,
        double const Ne
    ) const;

    /**
     * @brief Run both double and DD power series, return the relative error.
     *
     * Computes |dd_value - dbl_value| / (|dd_value| + 1e-300).
     * Useful for checking whether double precision is sufficient for a
     * given set of parameters.
     */
    double sigma_E_precision_check(
        double const E,
        double const E_prime,
        double const xi,
        double const tau,
        double const Ne
    ) const;

private:
    template<typename T>
    SeriesResult power_series(
        double const gamma,
        double const gamma_p,
        double const xi,
        double const tau,
        double const E,
        double const Ne
    ) const;

    SeriesResult asymptotic_series(
        double const gamma,
        double const gamma_p,
        double const xi,
        double const tau,
        double const E,
        double const Ne
    ) const;

    SeriesMethod method_;
    double eps_rel_;
    int n_min_, n_max_;

};

} // namespace compton

#endif
