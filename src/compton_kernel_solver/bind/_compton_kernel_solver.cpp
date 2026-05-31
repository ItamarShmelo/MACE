#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_kernel_solver/compton_kernel_solver.hpp"

namespace py = pybind11;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_solver, m) {
    m.doc() = "Adaptive Compton kernel solver: asymptotic or power series dispatch";

    py::module_::import("_compton_common");

    py::class_<ComptonKernelSolver>(m, "ComptonKernelSolver")
        .def(py::init<>())
        .def("sigma_E", &ComptonKernelSolver::sigma_E,
             py::arg("E"), py::arg("E_prime"), py::arg("xi"),
             py::arg("tau"), py::arg("Ne"))
        .def("sigma_E_vec", [](ComptonKernelSolver const& self,
                               double E,
                               py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                               double xi, double tau, double Ne) {
            auto in = E_prime_arr.unchecked<1>();
            py::ssize_t const n = in.shape(0);

            py::array_t<double> values(n);
            py::array_t<double> errors(n);

            auto v = values.mutable_unchecked<1>();
            auto e = errors.mutable_unchecked<1>();

            for (py::ssize_t i = 0; i < n; ++i) {
                SigmaResult r = self.sigma_E(E, in(i), xi, tau, Ne);
                v(i) = r.value;
                e(i) = r.estimated_abs_error;
            }
            return py::make_tuple(values, errors);
        }, py::arg("E"), py::arg("E_prime_arr"), py::arg("xi"),
           py::arg("tau"), py::arg("Ne"));
}
