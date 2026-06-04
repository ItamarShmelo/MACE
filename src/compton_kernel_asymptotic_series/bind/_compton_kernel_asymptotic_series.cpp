#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"
#include "compton_common/compton_common.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_asymptotic_series, m) {
    m.doc() = "Compton scattering kernel via low-temperature asymptotic series";

    py::module_::import("_compton_common");

    py::class_<ComptonKernelAsymptoticSeries>(m, "ComptonKernelAsymptoticSeries")
        .def(py::init<double, int, int>(),
             "eps_rel"_a = 1e-12,
             "n_min"_a = 4,
             "n_max"_a = 200)
        .def("sigma_E", &ComptonKernelAsymptoticSeries::sigma_E,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("sigma_E_vec", [](ComptonKernelAsymptoticSeries const& self,
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
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a)
        .def("dsigma_E_dT", &ComptonKernelAsymptoticSeries::dsigma_E_dT,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("dsigma_E_dT_vec", [](ComptonKernelAsymptoticSeries const& self,
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
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a);
}
