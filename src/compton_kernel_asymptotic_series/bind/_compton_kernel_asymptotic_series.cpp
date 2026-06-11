#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_common/bind_helpers.hpp"
#include "compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_asymptotic_series, m) {
    m.doc() = "Compton scattering kernel via low-temperature asymptotic series";

    py::module_::import("_compton_common");

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
}
