#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_common/bind_helpers.hpp"
#include "compton_kernel_solver/compton_kernel_solver.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_solver, m) {
    m.doc() = "Adaptive dispatch Compton kernel (asymptotic / double PS / Q64 / DD)";

    py::module_::import("_compton_common");

    py::class_<ComptonKernelSolver>(m, "ComptonKernelSolver")
        .def(py::init<double, double, double, double>(),
             "asymp_tau_alpha_threshold"_a   = 0.025,
             "gamma_double_precision_safe"_a = 0.02,
             "quadrature_self_tol"_a         = 1e-6,
             "asymp_gamma_dd_threshold"_a    = 0.002)
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
