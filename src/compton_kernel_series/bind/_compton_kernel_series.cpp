#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_kernel_series/compton_kernel_series.hpp"

namespace py = pybind11;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_series, m) {
    m.doc() = "Compton scattering kernel via Section 4 series expansions";

    py::enum_<SeriesMethod>(m, "SeriesMethod")
        .value("PowerSeries", SeriesMethod::PowerSeries)
        .value("Asymptotic", SeriesMethod::Asymptotic)
        .value("Auto", SeriesMethod::Auto);

    py::class_<SeriesResult>(m, "SeriesResult")
        .def_readonly("value", &SeriesResult::value)
        .def_readonly("estimated_abs_error", &SeriesResult::estimated_abs_error)
        .def_readonly("estimated_rel_error", &SeriesResult::estimated_rel_error)
        .def_readonly("terms_used", &SeriesResult::terms_used)
        .def_readonly("method_used", &SeriesResult::method_used)
        .def_readonly("converged", &SeriesResult::converged);

    py::class_<ComptonKernelSeries>(m, "ComptonKernelSeries")
        .def(py::init<SeriesMethod, double, int, int>(),
             py::arg("method") = SeriesMethod::Auto,
             py::arg("eps_rel") = 1e-12,
             py::arg("n_min") = 4,
             py::arg("n_max") = 200)
        .def("sigma_E", &ComptonKernelSeries::sigma_E,
             py::arg("E"), py::arg("E_prime"), py::arg("xi"),
             py::arg("tau"), py::arg("Ne"))
        .def("sigma_E_vec", [](const ComptonKernelSeries& self,
                               double E,
                               py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                               double xi, double tau, double Ne) {
            auto in = E_prime_arr.unchecked<1>();
            const py::ssize_t n = in.shape(0);

            py::array_t<double> values(n);
            py::array_t<double> errors(n);
            py::array_t<int> terms(n);
            auto out_values = values.mutable_unchecked<1>();
            auto out_errors = errors.mutable_unchecked<1>();
            auto out_terms = terms.mutable_unchecked<1>();

            for (py::ssize_t i = 0; i < n; ++i) {
                SeriesResult r = self.sigma_E(E, in(i), xi, tau, Ne);
                out_values(i) = r.value;
                out_errors(i) = r.estimated_abs_error;
                out_terms(i) = r.terms_used;
            }
            return py::make_tuple(values, errors, terms);
        }, py::arg("E"), py::arg("E_prime_arr"), py::arg("xi"),
           py::arg("tau"), py::arg("Ne"));

    m.def("ehat_cf", [](int const m, double const x) {
        return ehat_cf(m, x);
    }, py::arg("m"), py::arg("x"),
          "Scaled exponential integral via continued fraction: Ehat_m(x) = exp(x) * E_m(x)");
}
