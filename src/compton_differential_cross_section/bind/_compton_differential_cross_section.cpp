#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "utilities/bind_helpers.hpp"
#include "compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.hpp"
#include "compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"
#include "compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "utilities/gauss_laguerre.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

PYBIND11_MODULE(_compton_differential_cross_section, m) {
    m.doc() = "Compton differential cross-section kernels: power series, asymptotic series, quadrature, and adaptive solver";

    py::module_::import("_compton_common");

    // --- Power Series ---
    py::class_<ComptonPowerSeries>(m, "ComptonPowerSeries")
        .def(py::init<bool, double, int, int>(),
             "high_precision"_a = false,
             "eps_rel"_a = 1e-12,
             "n_min"_a = 4,
             "n_max"_a = 200)
        .def("sigma_E", &ComptonPowerSeries::sigma_E,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("sigma_E_precision_check", &ComptonPowerSeries::sigma_E_precision_check,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("sigma_E_vec", [](ComptonPowerSeries const& self,
                               double E,
                               py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                               double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonPowerSeries::sigma_E);
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a)
        .def("dsigma_E_dT", &ComptonPowerSeries::dsigma_E_dT,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("dsigma_E_dT_vec", [](ComptonPowerSeries const& self,
                                    double E,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                                    double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonPowerSeries::dsigma_E_dT);
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a)
        .def("dsigma_E_dT_precision_check", &ComptonPowerSeries::dsigma_E_dT_precision_check,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a);

    m.def("ehat_cf", [](int const m_order, double const x) {
        return ehat(m_order, x);
    }, "m"_a, "x"_a,
          "Scaled exponential integral via continued fraction: Ehat_m(x) = exp(x) * E_m(x)");

    // --- Asymptotic Series ---
    py::class_<ComptonKernelAsymptoticSeries>(m, "ComptonKernelAsymptoticSeries")
        .def(py::init<bool, double, int, int>(),
             "high_precision"_a = false,
             "eps_rel"_a = 1e-12,
             "n_min"_a = 4,
             "n_max"_a = 200)
        .def("sigma_E", &ComptonKernelAsymptoticSeries::sigma_E,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("sigma_E_precision_check", &ComptonKernelAsymptoticSeries::sigma_E_precision_check,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("sigma_E_vec", [](ComptonKernelAsymptoticSeries const& self,
                               double E,
                               py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                               double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonKernelAsymptoticSeries::sigma_E);
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a)
        .def("dsigma_E_dT", &ComptonKernelAsymptoticSeries::dsigma_E_dT,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("dsigma_E_dT_vec", [](ComptonKernelAsymptoticSeries const& self,
                                    double E,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                                    double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonKernelAsymptoticSeries::dsigma_E_dT);
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a)
        .def("dsigma_E_dT_precision_check", &ComptonKernelAsymptoticSeries::dsigma_E_dT_precision_check,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a);

    // --- Quadrature ---
    py::enum_<QuadratureForm>(m, "QuadratureForm")
        .value("PostIBP", QuadratureForm::PostIntegrationByParts)
        .value("PreIBP", QuadratureForm::PreIntegrationByParts);

    py::class_<ComptonKernelQuadrature>(m, "ComptonKernelQuadrature")
        .def(py::init<int, QuadratureForm>(),
             "NL"_a = 64,
             "form"_a = QuadratureForm::PostIntegrationByParts)
        .def("sigma_E", &ComptonKernelQuadrature::sigma_E,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("sigma_E_vec", [](ComptonKernelQuadrature const& self,
                               double E,
                               py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                               double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonKernelQuadrature::sigma_E);
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a)
        .def("dsigma_E_dT", &ComptonKernelQuadrature::dsigma_E_dT,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("dsigma_E_dT_vec", [](ComptonKernelQuadrature const& self,
                                    double E,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                                    double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonKernelQuadrature::dsigma_E_dT);
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a);

    m.def("scaled_K2", &scaled_K2, "x"_a,
          "Returns kve(2, x) = exp(x) * K_2(x)");

    m.def("scaled_K1", &scaled_K1, "x"_a,
          "Returns kve(1, x) = exp(x) * K_1(x)");

    m.def("kappa_ratio", &kappa_ratio, "tau"_a,
          "Returns K_1(1/tau) / K_2(1/tau)");

    m.def("gauss_laguerre_rule", [](int N) {
        auto rule = compton::compute_gauss_laguerre(N);

        py::array_t<double> nodes(rule.nodes.size());
        py::array_t<double> weights(rule.weights.size());

        auto nodes_buf = nodes.mutable_unchecked<1>();
        auto weights_buf = weights.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(rule.nodes.size()); ++i) {
            nodes_buf(i) = rule.nodes[i];
            weights_buf(i) = rule.weights[i];
        }

        return py::make_tuple(nodes, weights);
    }, "N"_a, "Compute N-point Gauss-Laguerre nodes and weights");

    // --- Solver ---
    py::class_<ComptonKernelSolver>(m, "ComptonKernelSolver")
        .def(py::init<double, double, double, double, double, double>(),
             "asymp_tau_alpha_threshold"_a   = 0.025,
             "gamma_double_precision_safe"_a = 0.02,
             "quadrature_self_tol"_a         = 1e-6,
             "asymp_gamma_dd_threshold"_a    = 0.002,
             "asymp_self_tol"_a              = 1e-3,
             "asymp_gamma_dd_cross_val_threshold"_a = 1e-4)
        .def("sigma_E", &ComptonKernelSolver::sigma_E,
             "E"_a, "E_prime"_a, "xi"_a, "T"_a, "Ne"_a)
        .def("dsigma_E_dT", &ComptonKernelSolver::dsigma_E_dT,
             "E"_a, "E_prime"_a, "xi"_a, "T"_a, "Ne"_a)
        .def("sigma_E_vec", [](ComptonKernelSolver const& self,
                                double E,
                                py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                                double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonKernelSolver::sigma_E);
        }, "E"_a, "E_prime_arr"_a, "xi"_a, "T"_a, "Ne"_a)
        .def("dsigma_E_dT_vec", [](ComptonKernelSolver const& self,
                                    double E,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                                    double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonKernelSolver::dsigma_E_dT);
        }, "E"_a, "E_prime_arr"_a, "xi"_a, "T"_a, "Ne"_a);
}
