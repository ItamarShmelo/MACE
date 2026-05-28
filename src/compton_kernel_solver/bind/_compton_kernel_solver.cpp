#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_kernel_solver/compton_kernel_solver.hpp"

namespace py = pybind11;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_solver, m) {
    m.doc() = "Robust adaptive Compton kernel solver with cascade method selection";

    py::enum_<SolverMethod>(m, "SolverMethod")
        .value("Asymptotic", SolverMethod::Asymptotic)
        .value("PowerSeries", SolverMethod::PowerSeries)
        .value("Quadrature", SolverMethod::Quadrature);

    py::class_<SolverResult>(m, "SolverResult")
        .def_readonly("value", &SolverResult::value)
        .def_readonly("estimated_abs_error", &SolverResult::estimated_abs_error)
        .def_readonly("estimated_rel_error", &SolverResult::estimated_rel_error)
        .def_readonly("terms_used", &SolverResult::terms_used)
        .def_readonly("method_used", &SolverResult::method_used)
        .def_readonly("used_fallback", &SolverResult::used_fallback)
        .def_readonly("target_met", &SolverResult::target_met)
        .def_readonly("clamped", &SolverResult::clamped)
        .def_readonly("tau_alpha_max", &SolverResult::tau_alpha_max)
        .def_readonly("conditioning", &SolverResult::conditioning);

    py::class_<ComptonKernelSolver>(m, "ComptonKernelSolver")
        .def(py::init<double, double>(),
             py::arg("target_rel_tol") = 1e-8,
             py::arg("target_abs_tol") = 1e-300)
        .def("sigma_E", &ComptonKernelSolver::sigma_E,
             py::arg("E"), py::arg("E_prime"), py::arg("xi"),
             py::arg("tau"), py::arg("Ne"))
        .def("sigma_E_vec", [](const ComptonKernelSolver& self,
                               double E,
                               py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                               double xi, double tau, double Ne) {
            auto in = E_prime_arr.unchecked<1>();
            const py::ssize_t n = in.shape(0);

            py::array_t<double> values(n);
            py::array_t<double> errors(n);
            py::array_t<int> methods(n);
            py::array_t<int> terms(n);
            py::array_t<int> fallbacks(n);
            py::array_t<int> target_mets(n);

            auto v = values.mutable_unchecked<1>();
            auto e = errors.mutable_unchecked<1>();
            auto met = methods.mutable_unchecked<1>();
            auto t = terms.mutable_unchecked<1>();
            auto fb = fallbacks.mutable_unchecked<1>();
            auto tm = target_mets.mutable_unchecked<1>();

            for (py::ssize_t i = 0; i < n; ++i) {
                SolverResult r = self.sigma_E(E, in(i), xi, tau, Ne);
                v(i) = r.value;
                e(i) = r.estimated_abs_error;
                met(i) = static_cast<int>(r.method_used);
                t(i) = r.terms_used;
                fb(i) = r.used_fallback ? 1 : 0;
                tm(i) = r.target_met ? 1 : 0;
            }
            return py::make_tuple(values, errors, methods, terms, fallbacks, target_mets);
        }, py::arg("E"), py::arg("E_prime_arr"), py::arg("xi"),
           py::arg("tau"), py::arg("Ne"));
}
