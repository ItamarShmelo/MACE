#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "compton_common/compton_common.hpp"
#include "compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"
#include "compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.hpp"
#include "compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.hpp"
#include "compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.hpp"
#include "compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "utilities/bind_helpers.hpp"
#include "utilities/gauss_laguerre.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

PYBIND11_MODULE(
    _compton_differential_cross_section,
    m) // NOLINT(misc-include-cleaner)
{
    m.doc() = "Compton differential cross-section kernels: power series, "
              "asymptotic series, quadrature, and adaptive solver";

    py::module_::import("compton_matrix._compton_common");

    // --- Power Series ---
    py::class_<ComptonPowerSeries>(m, "ComptonPowerSeries")
        .def(
            py::init<bool, double, int, int>(),
            "high_precision"_a = false, // NOLINT(misc-include-cleaner)
            "eps_rel"_a = 1e-8,
            "n_min"_a = 4,
            "n_max"_a = 500)
        .def(
            "sigma_E",
            &ComptonPowerSeries::sigma_E,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "sigma_E_precision_check",
            &ComptonPowerSeries::sigma_E_precision_check,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "sigma_E_vec",
            [](ComptonPowerSeries const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonPowerSeries::sigma_E);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT",
            &ComptonPowerSeries::dsigma_E_dT,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT_vec",
            [](ComptonPowerSeries const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonPowerSeries::dsigma_E_dT);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT_precision_check",
            &ComptonPowerSeries::dsigma_E_dT_precision_check,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a);

    m.def(
        "ehat_cf",
        [](int const m_order, double const x) { return ehat(m_order, x); },
        "m"_a,
        "x"_a,
        "Scaled exponential integral via continued fraction: Ehat_m(x) = "
        "exp(x) * E_m(x)");

    // --- Asymptotic Series ---
    py::class_<ComptonKernelAsymptoticSeries>(
        m,
        "ComptonKernelAsymptoticSeries")
        .def(
            py::init<bool, double, int, int>(),
            "high_precision"_a = false,
            "eps_rel"_a = 1e-8,
            "n_min"_a = 4,
            "n_max"_a = 200)
        .def(
            "sigma_E",
            &ComptonKernelAsymptoticSeries::sigma_E,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "sigma_E_precision_check",
            &ComptonKernelAsymptoticSeries::sigma_E_precision_check,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "sigma_E_vec",
            [](ComptonKernelAsymptoticSeries const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelAsymptoticSeries::sigma_E);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT",
            &ComptonKernelAsymptoticSeries::dsigma_E_dT,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT_vec",
            [](ComptonKernelAsymptoticSeries const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelAsymptoticSeries::dsigma_E_dT);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT_precision_check",
            &ComptonKernelAsymptoticSeries::dsigma_E_dT_precision_check,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a);

    // --- Quadrature ---
    py::enum_<QuadratureForm>(m, "QuadratureForm")
        .value("PostIBP", QuadratureForm::PostIntegrationByParts)
        .value("PreIBP", QuadratureForm::PreIntegrationByParts);

    py::class_<ComptonKernelQuadrature>(m, "ComptonKernelQuadrature")
        .def(
            py::init<int, QuadratureForm>(),
            "NL"_a = 64,
            "form"_a = QuadratureForm::PostIntegrationByParts)
        .def(
            "sigma_E",
            &ComptonKernelQuadrature::sigma_E,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "sigma_E_vec",
            [](ComptonKernelQuadrature const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelQuadrature::sigma_E);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT",
            &ComptonKernelQuadrature::dsigma_E_dT,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT_vec",
            [](ComptonKernelQuadrature const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelQuadrature::dsigma_E_dT);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a);

    m.def(
        "scaled_K2",
        &scaled_K2,
        "x"_a,
        "Returns kve(2, x) = exp(x) * K_2(x)");

    m.def(
        "scaled_K1",
        &scaled_K1,
        "x"_a,
        "Returns kve(1, x) = exp(x) * K_1(x)");

    m.def(
        "kappa_ratio",
        &kappa_ratio,
        "tau"_a,
        "Returns K_1(1/tau) / K_2(1/tau)");

    m.def(
        "gauss_laguerre_rule",
        [](int N) {
            auto rule = compton::compute_gauss_laguerre(N);

            py::array_t<double> nodes(static_cast<py::ssize_t>(
                rule.nodes.size())); // NOLINT(misc-include-cleaner)
            py::array_t<double> weights(
                static_cast<py::ssize_t>(rule.weights.size()));

            auto nodes_buf = nodes.mutable_unchecked<1>();
            auto weights_buf = weights.mutable_unchecked<1>();

            for (py::ssize_t i = 0; // NOLINT(misc-include-cleaner)
                 i < static_cast<py::ssize_t>(rule.nodes.size());
                 ++i) {
                nodes_buf(i) = rule.nodes[i];
                weights_buf(i) = rule.weights[i];
            }

            return py::make_tuple(
                nodes,
                weights); // NOLINT(misc-include-cleaner)
        },
        "N"_a,
        "Compute N-point Gauss-Laguerre nodes and weights");

    // --- Solver ---
    py::class_<ComptonKernelSolver>(m, "ComptonKernelSolver")
        .def(
            py::init<double, double, double, double, double, bool>(),
            "asymp_tau_alpha_threshold"_a =
                constants::ASYMP_TAU_ALPHA_THRESHOLD,
            "power_series_self_tol"_a = 1e-7,
            "asymp_self_tol"_a = 1e-7,
            "dd_power_series_self_tol"_a = 0.5,
            "dd_asymp_self_tol"_a = 0.5,
            "verbose"_a = false)
        .def(
            "sigma_E",
            &ComptonKernelSolver::sigma_E,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT",
            &ComptonKernelSolver::dsigma_E_dT,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "sigma_E_vec",
            [](ComptonKernelSolver const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelSolver::sigma_E);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT_vec",
            [](ComptonKernelSolver const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelSolver::dsigma_E_dT);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a);

    // --- Approximate (fifth-order global approximation) ---
    py::class_<ComptonKernelApproximate>(m, "ComptonKernelApproximate")
        .def(py::init<>())
        .def(
            "sigma_E",
            &ComptonKernelApproximate::sigma_E,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "sigma_E_vec",
            [](ComptonKernelApproximate const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelApproximate::sigma_E);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a);

    // --- Approximate Solver (KG5 + full solver fallback) ---
    py::class_<ComptonKernelApproximateSolver>(m, "ComptonKernelApproximateSolver")
        .def(
            py::init<double, double, double, double, double, double, double, bool>(),
            "gamma_tau_ratio"_a = 3.0,
            "tau_max"_a = 0.098,
            "asymp_tau_alpha_threshold"_a =
                constants::ASYMP_TAU_ALPHA_THRESHOLD,
            "power_series_self_tol"_a = 1e-7,
            "asymp_self_tol"_a = 1e-7,
            "dd_power_series_self_tol"_a = 0.5,
            "dd_asymp_self_tol"_a = 0.5,
            "verbose"_a = false)
        .def(
            "sigma_E",
            &ComptonKernelApproximateSolver::sigma_E,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT",
            &ComptonKernelApproximateSolver::dsigma_E_dT,
            "E"_a,
            "E_prime"_a,
            "xi"_a,
            "T"_a)
        .def(
            "sigma_E_vec",
            [](ComptonKernelApproximateSolver const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelApproximateSolver::sigma_E);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a)
        .def(
            "dsigma_E_dT_vec",
            [](ComptonKernelApproximateSolver const& self,
               double E,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> const&
                   E_prime_arr,
               double xi,
               double T) {
                return compton::bind::vectorize_sigma(
                    self,
                    E,
                    E_prime_arr,
                    xi,
                    T,
                    &ComptonKernelApproximateSolver::dsigma_E_dT);
            },
            "E"_a,
            "E_prime_arr"_a,
            "xi"_a,
            "T"_a);
}
