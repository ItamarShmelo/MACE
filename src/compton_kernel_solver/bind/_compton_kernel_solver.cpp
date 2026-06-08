#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
            auto in = E_prime_arr.unchecked<1>();
            py::ssize_t const n = in.shape(0);

            py::array_t<double> values(n);
            py::array_t<double> errors(n);
            auto out_values = values.mutable_unchecked<1>();
            auto out_errors = errors.mutable_unchecked<1>();

            for (py::ssize_t i = 0; i < n; ++i) {
                SigmaResult r = self.sigma_E(E, in(i), xi, T, Ne);
                out_values(i) = r.value;
                out_errors(i) = r.estimated_abs_error;
            }
            return py::make_tuple(values, errors);
        }, "E"_a, "E_prime_arr"_a, "xi"_a, "T"_a, "Ne"_a)
        .def("dsigma_E_dT_vec", [](ComptonKernelSolver const& self,
                                    double E,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                                    double xi, double T, double Ne) {
            auto in = E_prime_arr.unchecked<1>();
            py::ssize_t const n = in.shape(0);

            py::array_t<double> values(n);
            py::array_t<double> errors(n);
            auto out_values = values.mutable_unchecked<1>();
            auto out_errors = errors.mutable_unchecked<1>();

            for (py::ssize_t i = 0; i < n; ++i) {
                SigmaResult r = self.dsigma_E_dT(E, in(i), xi, T, Ne);
                out_values(i) = r.value;
                out_errors(i) = r.estimated_abs_error;
            }
            return py::make_tuple(values, errors);
        }, "E"_a, "E_prime_arr"_a, "xi"_a, "T"_a, "Ne"_a);
}
