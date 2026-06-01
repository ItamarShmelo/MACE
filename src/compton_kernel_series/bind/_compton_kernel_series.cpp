#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_kernel_series/compton_kernel_series.hpp"
#include "compton_common/compton_common.hpp"

namespace py = pybind11;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_series, m) {
    m.doc() = "Compton scattering kernel via Section 4 series expansions";

    py::module_::import("_compton_common");

    py::enum_<SeriesMethod>(m, "SeriesMethod")
        .value("PowerSeries", SeriesMethod::PowerSeries)
        .value("PowerSeriesHighPrecision", SeriesMethod::PowerSeriesHighPrecision)
        .value("Asymptotic", SeriesMethod::Asymptotic)
        .value("Auto", SeriesMethod::Auto);

    py::class_<ComptonKernelSeries>(m, "ComptonKernelSeries")
        .def(py::init<SeriesMethod, double, int, int>(),
             py::arg("method") = SeriesMethod::Auto,
             py::arg("eps_rel") = 1e-12,
             py::arg("n_min") = 4,
             py::arg("n_max") = 200)
        .def("sigma_E", &ComptonKernelSeries::sigma_E,
             py::arg("E"), py::arg("E_prime"), py::arg("xi"),
             py::arg("T"), py::arg("Ne"))
        .def("sigma_E_precision_check", &ComptonKernelSeries::sigma_E_precision_check,
             py::arg("E"), py::arg("E_prime"), py::arg("xi"),
             py::arg("T"), py::arg("Ne"))
        .def("sigma_E_vec", [](ComptonKernelSeries const& self,
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
        }, py::arg("E"), py::arg("E_prime_arr"), py::arg("xi"),
           py::arg("T"), py::arg("Ne"));

    m.def("ehat_cf", [](int const m, double const x) {
        return ehat(m, x);
    }, py::arg("m"), py::arg("x"),
          "Scaled exponential integral via continued fraction: Ehat_m(x) = exp(x) * E_m(x)");
}
