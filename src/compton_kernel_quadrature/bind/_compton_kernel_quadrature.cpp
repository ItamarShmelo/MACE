#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "compton_kernel_quadrature/gauss_laguerre.hpp"

namespace py = pybind11;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_quadrature, m) {
    m.doc() = "Compton scattering kernel via direct Gauss-Laguerre quadrature";

    py::module_::import("_compton_common");

    py::enum_<QuadratureForm>(m, "QuadratureForm")
        .value("PostIBP", QuadratureForm::PostIntegrationByParts)
        .value("PreIBP", QuadratureForm::PreIntegrationByParts);

    py::class_<ComptonKernelQuadrature>(m, "ComptonKernelQuadrature")
        .def(py::init<int, QuadratureForm>(),
             py::arg("NL") = 64,
             py::arg("form") = QuadratureForm::PostIntegrationByParts)
        .def("sigma_E", &ComptonKernelQuadrature::sigma_E,
             py::arg("E"), py::arg("E_prime"), py::arg("xi"),
             py::arg("tau"), py::arg("Ne"))
        .def("sigma_E_vec", [](const ComptonKernelQuadrature& self,
                               double E,
                               py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                               double xi, double tau, double Ne) {
            auto in = E_prime_arr.unchecked<1>();
            const py::ssize_t n = in.shape(0);

            py::array_t<double> values(n);
            py::array_t<double> errors(n);
            auto out_values = values.mutable_unchecked<1>();
            auto out_errors = errors.mutable_unchecked<1>();

            for (py::ssize_t i = 0; i < n; ++i) {
                SigmaResult r = self.sigma_E(E, in(i), xi, tau, Ne);
                out_values(i) = r.value;
                out_errors(i) = r.estimated_abs_error;
            }
            return py::make_tuple(values, errors);
        }, py::arg("E"), py::arg("E_prime_arr"), py::arg("xi"),
           py::arg("tau"), py::arg("Ne"))
        .def("dsigma_E_dtau", &ComptonKernelQuadrature::dsigma_E_dtau,
             py::arg("E"), py::arg("E_prime"), py::arg("xi"),
             py::arg("tau"), py::arg("Ne"))
        .def("dsigma_E_dtau_vec", [](const ComptonKernelQuadrature& self,
                                      double E,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                                      double xi, double tau, double Ne) {
            auto in = E_prime_arr.unchecked<1>();
            const py::ssize_t n = in.shape(0);

            py::array_t<double> values(n);
            py::array_t<double> errors(n);
            auto out_values = values.mutable_unchecked<1>();
            auto out_errors = errors.mutable_unchecked<1>();

            for (py::ssize_t i = 0; i < n; ++i) {
                SigmaResult r = self.dsigma_E_dtau(E, in(i), xi, tau, Ne);
                out_values(i) = r.value;
                out_errors(i) = r.estimated_abs_error;
            }
            return py::make_tuple(values, errors);
        }, py::arg("E"), py::arg("E_prime_arr"), py::arg("xi"),
           py::arg("tau"), py::arg("Ne"));

    m.def("scaled_K2", &scaled_K2, py::arg("x"),
          "Returns kve(2, x) = exp(x) * K_2(x)");

    m.def("scaled_K1", &scaled_K1, py::arg("x"),
          "Returns kve(1, x) = exp(x) * K_1(x)");

    m.def("kappa_ratio", &kappa_ratio, py::arg("tau"),
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
    }, py::arg("N"), "Compute N-point Gauss-Laguerre nodes and weights");
}
